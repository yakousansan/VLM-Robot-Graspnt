#include "safety_checker.h"

#include <cmath>

SafetyChecker::SafetyChecker(SafetyConfig config) : config_(config) {}

bool SafetyChecker::Validate(const GraspCommand& command, std::string& error) const {
    if (command.version != 1) {
        error = "unsupported protocol version";
        return false;
    }
    if (command.frame != "base") {
        error = "frame must be base";
        return false;
    }
    if (command.unit != "m_rad") {
        error = "unit must be m_rad";
        return false;
    }
    if (!std::isfinite(command.speed) || command.speed <= 0.0 || command.speed > config_.max_speed) {
        error = "speed is outside allowed range";
        return false;
    }
    for (double joint : command.home_joint_deg) {
        if (!std::isfinite(joint)) {
            error = "home_joint_deg contains a non-finite value";
            return false;
        }
    }
    return ValidatePose(command.pre_grasp_pose, "pre_grasp_pose", error)
        && ValidatePose(command.grasp_pose, "grasp_pose", error)
        && ValidatePose(command.lift_pose, "lift_pose", error);
}

bool SafetyChecker::ValidatePose(const Pose6& pose, const char* label, std::string& error) const {
    for (double value : pose) {
        if (!std::isfinite(value)) {
            error = std::string(label) + " contains a non-finite value";
            return false;
        }
    }
    const double x = pose[0];
    const double y = pose[1];
    const double z = pose[2];
    if (z < config_.min_grasp_z) {
        error = std::string(label) + " z is below min_grasp_z";
        return false;
    }
    if (x < config_.workspace.x_min || x > config_.workspace.x_max ||
        y < config_.workspace.y_min || y > config_.workspace.y_max ||
        z < config_.workspace.z_min || z > config_.workspace.z_max) {
        error = std::string(label) + " is outside workspace bounds";
        return false;
    }
    return true;
}

