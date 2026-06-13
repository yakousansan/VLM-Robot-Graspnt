#ifndef GRASPNT_ROBOT_EXECUTOR_SAFETY_CHECKER_H
#define GRASPNT_ROBOT_EXECUTOR_SAFETY_CHECKER_H

#include <string>

#include "protocol.h"

struct WorkspaceBounds {
    double x_min = -0.8;
    double x_max = 0.8;
    double y_min = -0.8;
    double y_max = 0.8;
    double z_min = 0.05;
    double z_max = 0.9;
};

struct SafetyConfig {
    double min_grasp_z = 0.05;
    double max_speed = 20.0;
    WorkspaceBounds workspace;
};

class SafetyChecker {
public:
    explicit SafetyChecker(SafetyConfig config);
    bool Validate(const GraspCommand& command, std::string& error) const;

private:
    bool ValidatePose(const Pose6& pose, const char* label, std::string& error) const;

    SafetyConfig config_;
};

#endif

