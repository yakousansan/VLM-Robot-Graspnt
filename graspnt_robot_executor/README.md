# GRASPNT Robot Executor

Independent Windows C++ UDP executor for the Python `graspnt_rm` pipeline.

This project is separate from `door4`. Use `door4` only as a reference for RealMan SDK, Modbus gripper, and Winsock patterns.

## Default Behavior

The executor connects to the RealMan controller, moves to the startup home joint pose, listens on UDP, and responds to `pose_request` with the current arm state.

The startup and post-grasp home joint pose is:

```text
[150, 0, 90, 0, -90, 90] deg
```

When a `grasp_execute` command is received, the executor validates and prints the command, replies with `ack`, and asks for local console confirmation before moving the robot.

## UDP

Default endpoint:

```text
127.0.0.1:6556
```

Expected Python config:

```yaml
execution:
  backend: "udp_cpp"
  udp_host: "127.0.0.1"
  udp_port: 6556
```

## RealMan SDK Integration

`src/robot_driver.cpp` is implemented with the same RealMan C++ SDK call pattern used by `door4/src/robot_control/robot_driver.cpp`.

The CMake file expects this SDK layout by default:

```text
graspnt_robot_executor/
  3rdparty/
    Robotic_Arm/
      include/
        rm_service.h
      lib/
        api_cpp.lib
        api_cpp.dll
```

You can override the SDK root in Visual Studio/CMake:

```powershell
-DROBOTIC_ARM_DIR=D:/path/to/Robotic_Arm
```

All SDK-specific headers and libraries stay inside `RobotDriver` so UDP, protocol, and safety code remain independent.
