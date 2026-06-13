from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for grasp video recording. "
            "Install it in the GRASPNT runtime environment."
        ) from exc
    return cv2


@dataclass
class GraspVideoRecorder:
    camera: Any
    path: str | None
    fps: float
    codec: str
    initial_frame: Any | None = None

    def __post_init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer: Any | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        if self.path is None:
            return
        if self._thread is not None:
            raise RuntimeError("grasp video recorder is already started")
        self._thread = threading.Thread(target=self._run, name="grasp-video-recorder")
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._error is not None:
            raise RuntimeError("grasp video recording failed") from self._error

    def _run(self) -> None:
        try:
            if self.initial_frame is not None:
                self._write_frame(self.initial_frame)
            interval = 1.0 / max(float(self.fps), 1.0)
            while not self._stop_event.is_set():
                try:
                    frame = self.camera.capture(warmup_frames=0)
                except StopIteration:
                    break
                self._write_frame(frame)
                time.sleep(interval)
        except BaseException as exc:
            self._error = exc

    def _write_frame(self, frame: Any) -> None:
        color = np.asarray(frame.color, dtype=np.uint8)
        if color.ndim != 3 or color.shape[2] != 3:
            raise ValueError("video frame color image must have shape HxWx3")
        if self._writer is None:
            cv2 = _import_cv2()
            height, width = color.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*str(self.codec)[:4].ljust(4))
            writer = cv2.VideoWriter(str(self.path), fourcc, float(self.fps), (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"failed to open grasp video writer: {self.path}")
            self._writer = writer
        self._writer.write(color)


def record_grasp_video(
    camera: Any,
    config: dict[str, Any] | None,
    initial_frame: Any | None = None,
) -> GraspVideoRecorder:
    config = config or {}
    if not bool(config.get("enabled", True)):
        return GraspVideoRecorder(
            camera=camera,
            path=None,
            fps=float(config.get("fps", 30)),
            codec=str(config.get("codec", "mp4v")),
            initial_frame=initial_frame,
        )

    output_dir = Path(config.get("output_dir", config.get("debug_dir", "debug_outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = str(config.get("extension", ".mp4"))
    if not extension.startswith("."):
        extension = f".{extension}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{timestamp}_grasp{extension}"
    return GraspVideoRecorder(
        camera=camera,
        path=str(path),
        fps=float(config.get("fps", 30)),
        codec=str(config.get("codec", "mp4v")),
        initial_frame=initial_frame,
    )
