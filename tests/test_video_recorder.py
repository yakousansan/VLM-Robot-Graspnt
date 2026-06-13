from types import SimpleNamespace

import numpy as np

from graspnt_vlm_rm.video_recorder import record_grasp_video


def test_record_grasp_video_disabled_returns_no_path():
    camera = SimpleNamespace(capture=lambda warmup_frames=0: None)

    recorder = record_grasp_video(camera, {"enabled": False})

    assert recorder.path is None


def test_record_grasp_video_writes_initial_frame_and_captured_frames(monkeypatch, tmp_path):
    written_frames = []
    released = []

    class FakeWriter:
        def __init__(self, path, fourcc, fps, size):
            self.path = path
            self.fourcc = fourcc
            self.fps = fps
            self.size = size

        def isOpened(self):
            return True

        def write(self, frame):
            written_frames.append(np.asarray(frame).copy())

        def release(self):
            released.append(True)

    class FakeCv2:
        @staticmethod
        def VideoWriter_fourcc(*chars):
            return "".join(chars)

        VideoWriter = FakeWriter

    frames = [
        SimpleNamespace(color=np.full((4, 5, 3), 2, dtype=np.uint8)),
        SimpleNamespace(color=np.full((4, 5, 3), 3, dtype=np.uint8)),
    ]

    def capture(warmup_frames=0):
        if frames:
            return frames.pop(0)
        raise StopIteration

    monkeypatch.setattr("graspnt_vlm_rm.video_recorder._import_cv2", lambda: FakeCv2)
    monkeypatch.setattr("graspnt_vlm_rm.video_recorder.time.sleep", lambda _seconds: None)

    recorder = record_grasp_video(
        SimpleNamespace(capture=capture),
        {
            "enabled": True,
            "output_dir": str(tmp_path),
            "fps": 30,
            "codec": "mp4v",
            "extension": ".mp4",
        },
        initial_frame=SimpleNamespace(color=np.full((4, 5, 3), 1, dtype=np.uint8)),
    )
    recorder.start()
    recorder.stop()

    assert recorder.path is not None
    assert recorder.path.endswith("_grasp.mp4")
    assert len(written_frames) >= 1
    assert written_frames[0].mean() == 1
    assert released == [True]
