#include "protocol.h"

#include <stdexcept>

template <std::size_t N>
static bool ReadArray(const nlohmann::json& message, const char* key, std::array<double, N>& out, std::string& error) {
    if (!message.contains(key) || !message.at(key).is_array() || message.at(key).size() != N) {
        error = std::string(key) + " must be an array with " + std::to_string(N) + " values";
        return false;
    }
    for (std::size_t i = 0; i < N; ++i) {
        if (!message.at(key).at(i).is_number()) {
            error = std::string(key) + " contains a non-numeric value";
            return false;
        }
        out[i] = message.at(key).at(i).get<double>();
    }
    return true;
}

bool ParseGraspCommand(const nlohmann::json& message, GraspCommand& command, std::string& error) {
    if (!message.is_object()) {
        error = "message must be a JSON object";
        return false;
    }
    if (message.value("type", "") != "grasp_execute") {
        error = "type must be grasp_execute";
        return false;
    }

    command.version = message.value("version", 1);
    command.seq = message.value("seq", 0);
    command.command_id = message.value("command_id", "");
    command.frame = message.value("frame", "");
    command.unit = message.value("unit", "");
    command.speed = message.value("speed", command.speed);
    command.score = message.value("score", 0.0);
    command.width = message.value("width", 0.0);

    if (command.command_id.empty()) {
        error = "command_id is required";
        return false;
    }
    if (message.contains("home_joint_deg") && !ReadArray(message, "home_joint_deg", command.home_joint_deg, error)) return false;
    if (!ReadArray(message, "pre_grasp_pose", command.pre_grasp_pose, error)) return false;
    if (!ReadArray(message, "grasp_pose", command.grasp_pose, error)) return false;
    if (!ReadArray(message, "lift_pose", command.lift_pose, error)) return false;
    return true;
}

nlohmann::json MakePoseResponse(int seq, const RobotState& state) {
    return {
        {"version", 1},
        {"type", "pose_response"},
        {"seq", seq},
        {"status", "ok"},
        {"frame", "base"},
        {"unit", "m_rad"},
        {"end_pose", state.end_pose},
        {"joint_deg", state.joint_deg},
        {"H_end2base", state.h_end2base},
    };
}

nlohmann::json MakeAck(int seq, const std::string& command_id, const std::string& status, const std::string& reason) {
    nlohmann::json response = {
        {"version", 1},
        {"type", "ack"},
        {"seq", seq},
        {"command_id", command_id},
        {"status", status},
    };
    if (!reason.empty()) {
        response["reason"] = reason;
    }
    return response;
}

nlohmann::json MakeResult(int seq, const std::string& command_id, const std::string& status, const std::string& reason) {
    nlohmann::json response = {
        {"version", 1},
        {"type", "result"},
        {"seq", seq},
        {"command_id", command_id},
        {"status", status},
    };
    if (!reason.empty()) {
        response["reason"] = reason;
    }
    return response;
}
