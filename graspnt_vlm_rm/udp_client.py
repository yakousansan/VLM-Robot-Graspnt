from __future__ import annotations

import json
import socket
from datetime import datetime
from typing import Any


def build_pose_request(seq: int) -> dict[str, Any]:
    return {
        "version": 1,
        "type": "pose_request",
        "seq": int(seq),
    }


def build_grasp_execute_request(
    seq: int,
    command_id: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": 1,
        "type": "grasp_execute",
        "seq": int(seq),
        "command_id": command_id,
        "frame": "base",
        "unit": "m_rad",
        "pre_grasp_pose": list(plan["pre_grasp_pose"]),
        "grasp_pose": list(plan["grasp_pose"]),
        "lift_pose": list(plan["lift_pose"]),
        "score": float(plan["score"]),
        "width": float(plan["width"]),
    }


def extract_current_end_pose(response: dict[str, Any]) -> list[float]:
    if response.get("type") != "pose_response":
        raise RuntimeError(f"expected pose_response, got {response.get('type')}")
    if response.get("status") != "ok":
        reason = response.get("reason", "unknown error")
        raise RuntimeError(f"pose_response failed: {reason}")
    end_pose = response.get("end_pose")
    if not isinstance(end_pose, list) or len(end_pose) != 6:
        raise RuntimeError("pose_response must include a 6-value end_pose")
    return [float(value) for value in end_pose]


class UdpRobotClient:
    def __init__(
        self,
        host: str,
        port: int,
        ack_timeout_sec: float = 1.0,
        result_timeout_sec: float = 60.0,
        max_retries: int = 3,
    ):
        self.host = host
        self.port = int(port)
        self.ack_timeout_sec = float(ack_timeout_sec)
        self.result_timeout_sec = float(result_timeout_sec)
        self.max_retries = int(max_retries)
        self._seq = 0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self) -> None:
        self._socket.close()

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _send(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        self._socket.sendto(payload, (self.host, self.port))

    def _receive(self, timeout_sec: float) -> dict[str, Any]:
        self._socket.settimeout(timeout_sec)
        payload, _address = self._socket.recvfrom(65535)
        try:
            response = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("received invalid JSON from robot executor") from exc
        if not isinstance(response, dict):
            raise RuntimeError("robot executor response must be a JSON object")
        return response

    def request_pose(self) -> dict[str, Any]:
        seq = self._next_seq()
        request = build_pose_request(seq)
        last_timeout = None
        for _ in range(max(self.max_retries, 1)):
            self._send(request)
            try:
                response = self._receive(self.ack_timeout_sec)
            except socket.timeout as exc:
                last_timeout = exc
                continue
            if response.get("seq") != seq:
                continue
            extract_current_end_pose(response)
            return response
        raise TimeoutError("timed out waiting for pose_response") from last_timeout

    def execute_grasp(
        self,
        plan: dict[str, Any],
        command_id: str | None = None,
    ) -> dict[str, Any]:
        seq = self._next_seq()
        command_id = command_id or make_command_id(seq)
        request = build_grasp_execute_request(seq, command_id, plan)

        last_timeout = None
        accepted = False
        for _ in range(max(self.max_retries, 1)):
            self._send(request)
            try:
                response = self._receive(self.ack_timeout_sec)
            except socket.timeout as exc:
                last_timeout = exc
                continue
            if response.get("seq") != seq or response.get("command_id") != command_id:
                continue
            if response.get("type") != "ack":
                raise RuntimeError(f"expected ack, got {response.get('type')}")
            if response.get("status") != "accepted":
                reason = response.get("reason", "unknown error")
                raise RuntimeError(f"grasp command rejected: {reason}")
            accepted = True
            break

        if not accepted:
            raise TimeoutError("timed out waiting for grasp ack") from last_timeout

        response = self._receive(self.result_timeout_sec)
        if response.get("seq") != seq or response.get("command_id") != command_id:
            raise RuntimeError("received result for a different grasp command")
        if response.get("type") != "result":
            raise RuntimeError(f"expected result, got {response.get('type')}")
        if response.get("status") == "cancelled":
            return response
        if response.get("status") != "success":
            reason = response.get("reason", "unknown error")
            raise RuntimeError(f"grasp execution failed: {reason}")
        return response


def make_command_id(seq: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{int(seq):04d}"
