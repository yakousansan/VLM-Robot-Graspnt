#ifndef GRASPNT_ROBOT_EXECUTOR_PROTOCOL_H
#define GRASPNT_ROBOT_EXECUTOR_PROTOCOL_H

#include <array>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using Pose6 = std::array<double, 6>;
using Joint6 = std::array<double, 6>;
using Matrix4 = std::array<std::array<double, 4>, 4>;

inline constexpr Joint6 kDefaultHomeJointDeg{140.0, 0.0, 90.0, 0.0, -90.0, 90.0};
inline constexpr double kDefaultMoveSpeed = 10.0;

struct RobotState {
    Pose6 end_pose{};
    Joint6 joint_deg{};
    Matrix4 h_end2base{};
};

struct GraspCommand {
    int version = 1;
    int seq = 0;
    std::string command_id;
    std::string frame;
    std::string unit;
    double speed = kDefaultMoveSpeed;
    Joint6 home_joint_deg = kDefaultHomeJointDeg;
    Pose6 pre_grasp_pose{};
    Pose6 grasp_pose{};
    Pose6 lift_pose{};
    double score = 0.0;
    double width = 0.0;
};

bool ParseGraspCommand(const nlohmann::json& message, GraspCommand& command, std::string& error);
nlohmann::json MakePoseResponse(int seq, const RobotState& state);
nlohmann::json MakeAck(int seq, const std::string& command_id, const std::string& status, const std::string& reason = "");
nlohmann::json MakeResult(int seq, const std::string& command_id, const std::string& status, const std::string& reason = "");

#endif
