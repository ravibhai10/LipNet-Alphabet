"""Tests for app/services/."""
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

class TestDatasetService:
    def test_invalid_letter_rejected(self):
        from app.services.dataset_service import DatasetService
        ds = DatasetService(dataset_root=Path(tempfile.mkdtemp()))
        r = ds.record(letter="@", speaker_id="s", session_id="x", show_preview=False)
        assert not r.success and "Invalid letter" in r.error
    def test_empty_speaker_rejected(self):
        from app.services.dataset_service import DatasetService
        ds = DatasetService(dataset_root=Path(tempfile.mkdtemp()))
        r = ds.record(letter="A", speaker_id="", session_id="x", show_preview=False)
        assert not r.success and "speaker_id" in r.error.lower()
    def test_empty_session_rejected(self):
        from app.services.dataset_service import DatasetService
        ds = DatasetService(dataset_root=Path(tempfile.mkdtemp()))
        r = ds.record(letter="A", speaker_id="s", session_id="", show_preview=False)
        assert not r.success and "session_id" in r.error.lower()
    def test_negative_duration_rejected(self):
        from app.services.dataset_service import DatasetService
        ds = DatasetService(dataset_root=Path(tempfile.mkdtemp()))
        r = ds.record(letter="A", speaker_id="s", session_id="x", duration_seconds=-1.0, show_preview=False)
        assert not r.success
    def test_get_stats_empty(self):
        from app.services.dataset_service import DatasetService
        ds = DatasetService(dataset_root=Path(tempfile.mkdtemp()))
        s = ds.get_stats()
        assert s["recordings"] == 0 and len(s["missing_letters"]) == 26
    def test_has_real_data_false_for_empty(self):
        from app.services.dataset_service import DatasetService
        ds = DatasetService(dataset_root=Path(tempfile.mkdtemp()))
        assert ds.has_real_data() is False
    def test_get_stats_with_recording(self):
        from app.services.dataset_service import DatasetService
        from data.schema import DATA_SOURCE_REAL_CAMERA
        tmp = Path(tempfile.mkdtemp())
        rd = tmp / "recordings" / "A" / "spk001" / "ses001" / "rec001"
        rd.mkdir(parents=True)
        (rd / "video.mp4").write_bytes(b"fake")
        (rd / "metadata.json").write_text(json.dumps({
            "recording_id": "rec001", "letter": "A", "speaker_id": "spk001",
            "session_id": "ses001", "data_source": DATA_SOURCE_REAL_CAMERA,
            "landmark_source": "mediapipe", "timestamp": "2024-01-01T00:00:00",
        }), encoding="utf-8")
        ds = DatasetService(dataset_root=tmp)
        s = ds.get_stats()
        assert s["recordings"] == 1 and "A" in s["letter_counts"]
    @patch("app.services.dataset_service.VideoIngestor")
    def test_record_mocked(self, MockIngestor):
        from app.services.dataset_service import DatasetService
        from data.schema import DATA_SOURCE_REAL_CAMERA
        m = MagicMock()
        m.recording_id = "rec_001"
        m.data_source = DATA_SOURCE_REAL_CAMERA
        MockIngestor.return_value.record_webcam.return_value = m
        ds = DatasetService(dataset_root=Path(tempfile.mkdtemp()))
        r = ds.record(letter="A", speaker_id="s", session_id="x", show_preview=False)
        assert r.success and r.metadata.data_source == DATA_SOURCE_REAL_CAMERA
    def test_synthetic_fails_validation(self):
        from app.services.dataset_service import DatasetService
        synth = Path(__file__).parents[1] / "synthetic_dataset"
        if not synth.exists(): pytest.skip("synthetic_dataset not present")
        ds = DatasetService(dataset_root=synth)
        assert ds.validate().has_critical_errors()

class TestTrainingService:
    def test_pre_check_fails_on_empty(self):
        from app.services.training_service import TrainingService
        c = TrainingService().pre_training_check(dataset_root=Path(tempfile.mkdtemp()))
        assert not c.passed and len(c.errors) > 0
    def test_pre_check_fails_on_synthetic(self):
        from app.services.training_service import TrainingService
        synth = Path(__file__).parents[1] / "synthetic_dataset"
        if not synth.exists(): pytest.skip("synthetic_dataset not present")
        c = TrainingService().pre_training_check(dataset_root=synth)
        assert not c.passed
    def test_list_checkpoints_returns_list(self):
        from app.services.training_service import TrainingService
        assert isinstance(TrainingService().list_checkpoints(), list)
    def test_get_checkpoint_info_returns_dict(self):
        from app.services.training_service import TrainingService
        assert isinstance(TrainingService().get_checkpoint_info(Path("runs/letter_decoder")), dict)
    def test_get_config_has_attrs(self):
        from app.services.training_service import TrainingService
        cfg = TrainingService().get_config()
        assert cfg is not None and hasattr(cfg, "training")

class TestInferenceService:
    def test_creation(self):
        from app.services.inference_service import InferenceService
        s = InferenceService()
        assert s._device is not None and isinstance(s._models, dict)
    def test_list_checkpoints_returns_list(self):
        from app.services.inference_service import InferenceService
        assert isinstance(InferenceService().list_checkpoints(), list)
    def test_load_raises_for_missing(self):
        from app.services.inference_service import InferenceService
        with pytest.raises(FileNotFoundError):
            InferenceService().load_checkpoint(Path(tempfile.mkdtemp()) / "x")
    def test_unload_all(self):
        from app.services.inference_service import InferenceService
        InferenceService().unload_all()
    def test_prediction_result_dataclass(self):
        from app.services.inference_service import PredictionResult
        r = PredictionResult(letter="A", confidence=0.95, state="CONFIDENT")
        assert r.letter == "A" and r.confidence == 0.95
    def test_get_checkpoint_summary_missing(self):
        from app.services.inference_service import InferenceService
        assert InferenceService().get_checkpoint_summary(Path(tempfile.mkdtemp())) == {}

class TestJobManager:
    def test_submit_creates_job(self):
        from app.services.job_manager import get_job_manager
        jm = get_job_manager()
        for _ in range(50):
            if not jm.is_running(): break
            time.sleep(0.05)
        jid = jm.submit(name="t", target=lambda: 42)
        job = jm.get(jid)
        assert job is not None and job.name == "t"
    def test_get_unknown_returns_none(self):
        from app.services.job_manager import get_job_manager
        assert get_job_manager().get("x_unk") is None
    def test_is_running_returns_bool(self):
        from app.services.job_manager import get_job_manager
        assert isinstance(get_job_manager().is_running(), bool)
    def test_cancel_unknown_returns_false(self):
        from app.services.job_manager import get_job_manager
        assert get_job_manager().cancel("x_unk") is False
    def test_latest_job_returns_last(self):
        from app.services.job_manager import get_job_manager
        jm = get_job_manager()
        for _ in range(50):
            if not jm.is_running(): break
            time.sleep(0.05)
        jm.submit(name="ja", target=lambda: time.sleep(0.01))
        for _ in range(50):
            if not jm.is_running(): break
            time.sleep(0.05)
        jm.submit(name="jb", target=lambda: time.sleep(0.01))
        latest = jm.latest_job()
        assert latest is not None and latest.name == "jb"
    def test_list_jobs_returns_list(self):
        from app.services.job_manager import get_job_manager
        assert isinstance(get_job_manager().list_jobs(), list)
