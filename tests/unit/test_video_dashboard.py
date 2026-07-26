from dms.pipeline.video_dashboard import VideoDashboard


def test_dashboard_accumulates_processed_frames_across_workers() -> None:
    dashboard = VideoDashboard(enabled=True)

    dashboard.handle_event({
        "pid": 100,
        "status": "running",
        "subject_id": "01",
        "video_name": "a.mov",
        "frames_processed": 10,
        "total_frames": 100,
        "progress_percent": 10.0,
        "fps": 10.0,
        "avg_inference_ms": 20.0,
        "elapsed_sec": 1.0,
        "eta_sec": 9.0,
    })
    dashboard.handle_event({
        "pid": 100,
        "status": "running",
        "subject_id": "01",
        "video_name": "a.mov",
        "frames_processed": 20,
        "total_frames": 100,
        "progress_percent": 20.0,
        "fps": 10.0,
        "avg_inference_ms": 20.0,
        "elapsed_sec": 2.0,
        "eta_sec": 8.0,
    })

    assert dashboard._overall_processed_frames == 20
