#ifndef GRASPNT_ROBOT_EXECUTOR_GRASP_EXECUTOR_H
#define GRASPNT_ROBOT_EXECUTOR_GRASP_EXECUTOR_H

#include <string>
#include <unordered_set>

#include "protocol.h"
#include "robot_driver.h"
#include "safety_checker.h"

class GraspExecutor {
public:
    GraspExecutor(RobotDriver& robot, SafetyChecker safety);

    bool Validate(const GraspCommand& command, std::string& error);
    bool Execute(const GraspCommand& command, std::string& error);

private:
    RobotDriver& robot_;
    SafetyChecker safety_;
    std::unordered_set<std::string> executed_commands_;
};

#endif
