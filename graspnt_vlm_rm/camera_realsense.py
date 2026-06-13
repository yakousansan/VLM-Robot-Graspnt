from dataclasses import dataclass

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    scale: float


@dataclass(frozen=True)
class RGBDFrame:
    color: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    depth_scale: float


class RealSenseCamera:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: str | None = None,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.pipeline = None
        self.config = None
        self.align = None
        self.profile = None
        self.depth_scale = None

    def start(self) -> None:
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is required to use RealSenseCamera. "
                "Install Intel RealSense SDK Python bindings before starting the camera."
            )

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if self.serial:
            self.config.enable_device(str(self.serial))
        self.config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        self.config.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps,
        )
        self.align = rs.align(rs.stream.color)
        self.profile = self.pipeline.start(self.config)
        self.depth_scale = (
            self.profile.get_device().first_depth_sensor().get_depth_scale()
        )
        if self.depth_scale <= 0:
            self.stop()
            raise RuntimeError(f"Invalid RealSense depth scale: {self.depth_scale}")

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        self.pipeline = None
        self.config = None
        self.align = None
        self.profile = None
        self.depth_scale = None

    def capture(self, warmup_frames: int = 5) -> RGBDFrame:
        if self.pipeline is None or self.align is None or self.depth_scale is None:
            raise RuntimeError("RealSenseCamera must be started before capture")
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative")

        aligned_frames = None
        for _ in range(warmup_frames + 1):
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense capture did not return color and depth frames")

        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        camera_intrinsics = CameraIntrinsics(
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.ppx,
            cy=intrinsics.ppy,
            scale=1.0 / self.depth_scale,
        )
        return RGBDFrame(
            color=np.asanyarray(color_frame.get_data()),
            depth=np.asanyarray(depth_frame.get_data()),
            intrinsics=camera_intrinsics,
            depth_scale=self.depth_scale,
        )
