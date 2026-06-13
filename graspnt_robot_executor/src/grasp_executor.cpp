#include "grasp_executor.h"

GraspExecutor::GraspExecutor(RobotDriver& robot, SafetyChecker safety)
    : robot_(robot), safety_(safety) {}

bool GraspExecutor::Validate(const GraspCommand& command, std::string& error) {
    if (executed_commands_.count(command.command_id) > 0) {
        error = "duplicate command_id";
        return false;
    }
    if (!safety_.Validate(command, error)) {
        return false;
    }
    if (!robot_.TargetReachable(command.pre_grasp_pose) ||
        !robot_.TargetReachable(command.grasp_pose) ||
        !robot_.TargetReachable(command.lift_pose)) {
        error = "target unreachable";
        return false;
    }
    return true;
}

bool GraspExecutor::Execute(const GraspCommand& command, std::string& error) {
    if (!Validate(command, error)) {
        return false;
    }
    executed_commands_.insert(command.command_id);
    if (!robot_.MoveJ(command.home_joint_deg, command.speed)) {
        error = "failed to move home";
        return false;
    }
    if (!robot_.ConfigureModbus()) {
        error = "failed to configure Modbus";
        return false;
    }
    if (!robot_.OpenGripper()) {
        error = "failed to open gripper";
        return false;
    }
    if (!robot_.MoveL(command.pre_grasp_pose, command.speed)) {
        error = "failed to move pre_grasp_pose";
        return false;
    }
    if (!robot_.MoveL(command.grasp_pose, command.speed)) {
        error = "failed to move grasp_pose";
        return false;
    }
    if (!robot_.CloseGripper()) {
        error = "failed to close gripper";
        return false;
    }
    if (!robot_.MoveL(command.lift_pose, command.speed)) {
        error = "failed to move lift_pose";
        return false;
    }
    if (!robot_.MoveJ(command.home_joint_deg, command.speed)) {
        error = "failed to return home";
        return false;
    }
    return true;
}
