"""Core pipeline that processes FL3D images through the full CV stack."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

from dms.analysis.eda import ExploratoryAnalyzer
from dms.dataset.loader import FL3DDatasetLoader
from dms.dataset.validator import DatasetValidator
from dms.detectors.factory import create_face_detector
from dms.domain.enums import DriverState, Split
from dms.domain.features import FeatureVector, ProcessedSample
from dms.domain.sample import DriverSample
from dms.evaluation.failure_analysis import FailureAnalyzer
from dms.evaluation.metrics import ModelEvaluator
from dms.features.base import FeatureContext
from dms.features.extractor import FeatureExtractor
from dms.landmarks.factory import create_landmark_extractor
from dms.landmarks.processor import LandmarkProcessor
from dms.landmarks.schema import LandmarkSchema
from dms.rules.engine import RuleEngine
from dms.utils.config import AppConfig, load_config, load_landmark_schema
from dms.utils.io import ensure_dir, save_dataframe, save_json
from dms.utils.logging import get_logger
from dms.utils.paths import get_project_root
from dms.visualization.plots import PlotGenerator

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Artifacts produced by a pipeline run."""

    run_dir: Path
    feature_dataset_path_csv: Path
    feature_dataset_path_parquet: Path
    evaluation_split: str
    samples_processed: int


class PipelineRunner:
    """Orchestrate dataset loading, CV processing, rules, and evaluation."""

    def __init__(self, config: AppConfig, project_root: Path | None = None) -> None:
        self._config = config
        self._root = project_root or get_project_root()
        self._schema = LandmarkSchema.from_dict(
            load_landmark_schema(config.landmarks.schema_path)
        )

    def run(self, splits: list[str] | None = None) -> PipelineResult:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = ensure_dir(self._config.outputs_dir_path(self._root) / "runs" / run_id)
        save_json(self._config.model_dump(), run_dir / "config_snapshot.json")

        split_names = splits or self._config.dataset.splits
        loader = FL3DDatasetLoader(self._config, self._root)
        samples_by_split = loader.load_splits(split_names)

        validator = DatasetValidator()
        validation_report = validator.validate_and_save(
            samples_by_split,
            run_dir / "validation_report.json",
        )
        logger.info(
            "Validation: %d valid / %d total samples",
            validation_report.valid_samples,
            validation_report.total_samples,
        )

        all_processed: list[ProcessedSample] = []
        with create_face_detector(self._config.detector) as detector, create_landmark_extractor(
            self._config.landmarks
        ) as landmark_extractor:
            landmark_processor = LandmarkProcessor(self._schema)
            feature_extractor = FeatureExtractor(self._config.features)
            rule_engine = RuleEngine(self._config.rules)

            for split_name, samples in samples_by_split.items():
                logger.info("Processing split '%s' (%d samples)", split_name, len(samples))
                for index, sample in enumerate(samples, start=1):
                    processed = self._process_sample(
                        sample=sample,
                        detector=detector,
                        landmark_extractor=landmark_extractor,
                        landmark_processor=landmark_processor,
                        feature_extractor=feature_extractor,
                        rule_engine=rule_engine,
                    )
                    all_processed.append(processed)

                    if index % 500 == 0:
                        logger.info("  %s: processed %d / %d", split_name, index, len(samples))

        df = pd.DataFrame([p.to_dataset_row() for p in all_processed])

        features_dir = ensure_dir(self._config.features_dir_path(self._root))
        csv_path = features_dir / f"features_{run_id}.csv"
        parquet_path = features_dir / f"features_{run_id}.parquet"
        save_dataframe(df, csv_path)
        save_dataframe(df, parquet_path)

        eval_split = self._config.evaluation.primary_split
        eval_df = df[df["split"] == eval_split] if ("split" in df.columns and (df["split"] == eval_split).any()) else df
        evaluator = ModelEvaluator()
        evaluation = evaluator.evaluate_dataframe(eval_df)
        save_json(evaluation.to_dict(), run_dir / "metrics" / "evaluation.json")

        failure_analyzer = FailureAnalyzer()
        if self._config.evaluation.generate_failure_analysis:
            failure_analyzer.save_report(eval_df, run_dir / "reports" / "failure_analysis.json")
            failure_analyzer.export_misclassified(
                eval_df,
                run_dir / "reports" / "misclassified.csv",
            )

        eda = ExploratoryAnalyzer()
        eda.save_report(df, run_dir / "reports" / "eda_summary.json")

        if self._config.analysis.generate_plots:
            plotter = PlotGenerator(run_dir / "figures")
            plotter.generate_all(eval_df, evaluation)

        logger.info("Pipeline complete. Run directory: %s", run_dir)
        return PipelineResult(
            run_dir=run_dir,
            feature_dataset_path_csv=csv_path,
            feature_dataset_path_parquet=parquet_path,
            evaluation_split=eval_split,
            samples_processed=len(all_processed),
        )

    def _process_sample(
        self,
        sample: DriverSample,
        detector,
        landmark_extractor,
        landmark_processor: LandmarkProcessor,
        feature_extractor: FeatureExtractor,
        rule_engine: RuleEngine,
    ) -> ProcessedSample:
        image_bgr = cv2.imread(str(sample.image_path))
        if image_bgr is None:
            return self._failed_sample(sample, "Failed to read image")

        detection = detector.detect(image_bgr)
        if not detection.detected:
            if self._config.pipeline.skip_on_detection_failure:
                return self._failed_sample(sample, "Face not detected")
            return self._failed_sample(sample, "Face not detected")

        landmarks = landmark_extractor.extract(image_bgr)
        if landmarks is None:
            return self._failed_sample(sample, "Landmark extraction failed")

        normalized = landmark_processor.process(landmarks)
        context = FeatureContext(
            landmarks=landmarks,
            normalized=normalized,
            schema=self._schema,
            image_bgr=image_bgr,
            detection=detection,
        )

        features = feature_extractor.extract(
            context=context,
            sample_id=sample.sample_id,
            subject_id=sample.subject_id,
        )
        prediction = rule_engine.predict(features)

        return ProcessedSample(
            sample=sample,
            features=features,
            predicted_label=prediction.predicted_label,
            detection_confidence=detection.confidence,
            landmark_confidence=landmarks.confidence,
            processing_success=True,
        )

    @staticmethod
    def _failed_sample(sample: DriverSample, message: str) -> ProcessedSample:
        return ProcessedSample(
            sample=sample,
            features=FeatureVector(
                sample_id=sample.sample_id,
                subject_id=sample.subject_id,
                features={},
            ),
            predicted_label=DriverState.ALERT,
            detection_confidence=0.0,
            landmark_confidence=0.0,
            processing_success=False,
            error_message=message,
        )


def run_from_config(config_path: str | Path | None = None, splits: list[str] | None = None) -> PipelineResult:
    config = load_config(config_path)
    runner = PipelineRunner(config)
    return runner.run(splits=splits)
