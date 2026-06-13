from types import SimpleNamespace

import numpy as np

from graspnt_vlm_rm import visualization


def test_save_debug_artifacts_does_not_write_grasp_json_by_default(
    monkeypatch,
    tmp_path,
):
    class FakeCv2:
        COLORMAP_JET = 2

        @staticmethod
        def imwrite(path, image):
            tmp_path.joinpath(path).parent.mkdir(parents=True, exist_ok=True)
            tmp_path.joinpath(path).write_bytes(b"image")
            return True

        @staticmethod
        def applyColorMap(image, _colormap):
            return np.dstack([image, image, image])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(visualization, "_import_cv2", lambda: FakeCv2)

    frame = SimpleNamespace(
        color=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.ones((4, 4), dtype=np.uint16),
        intrinsics=SimpleNamespace(fx=1, fy=1, cx=0, cy=0),
    )
    debug_data = SimpleNamespace(workspace_mask=np.ones((4, 4), dtype=bool))

    saved_files = visualization.save_debug_artifacts(
        frame,
        candidates=[],
        debug_data=debug_data,
        plan={},
        config={"debug_dir": "debug_outputs"},
    )

    assert len(saved_files) == 2
    assert not list(tmp_path.glob("debug_outputs/*_grasps.json"))


def test_save_debug_artifacts_does_not_write_point_cloud(monkeypatch, tmp_path):
    class FakeCv2:
        COLORMAP_JET = 2

        @staticmethod
        def imwrite(path, image):
            tmp_path.joinpath(path).parent.mkdir(parents=True, exist_ok=True)
            tmp_path.joinpath(path).write_bytes(b"image")
            return True

        @staticmethod
        def applyColorMap(image, _colormap):
            return np.dstack([image, image, image])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(visualization, "_import_cv2", lambda: FakeCv2)
    monkeypatch.setattr(
        visualization,
        "_import_open3d",
        lambda: (_ for _ in ()).throw(AssertionError("Open3D should not be imported")),
    )

    frame = SimpleNamespace(
        color=np.zeros((4, 4, 3), dtype=np.uint8),
        depth=np.ones((4, 4), dtype=np.uint16),
        intrinsics=SimpleNamespace(fx=1, fy=1, cx=0, cy=0),
    )
    debug_data = SimpleNamespace(
        workspace_mask=np.ones((4, 4), dtype=bool),
        cloud_points=np.zeros((1, 3), dtype=np.float32),
        cloud_colors=np.zeros((1, 3), dtype=np.float32),
    )

    saved_files = visualization.save_debug_artifacts(
        frame,
        candidates=[],
        debug_data=debug_data,
        plan={},
        config={"debug_dir": "debug_outputs"},
    )

    assert len(saved_files) == 2
    assert not list(tmp_path.glob("debug_outputs/*.ply"))
