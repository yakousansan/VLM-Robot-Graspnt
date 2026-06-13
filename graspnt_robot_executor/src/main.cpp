#include <iomanip>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "grasp_executor.h"
#include "protocol.h"
#include "robot_driver.h"
#include "safety_checker.h"
#include "udp_server.h"

static nlohmann::json BuildCompactLogJson(const nlohmann::json& message) {
    nlohmann::json compact = message;
    compact.erase("command_id");
    compact.erase("frame");
    compact.erase("H_end2base");
    compact.erase("end_pose");
    compact.erase("joint_deg");
    return compact;
}

static void LogUdpJson(const std::string& direction, const nlohmann::json& message) {
    if (message.value("type", "") == "pose_response" && message.value("status", "") == "ok") {
        if (message.contains("H_end2base")) {
            std::cout << "[UDP][" << direction << "] H_end2base="
                      << message.at("H_end2base").dump() << std::endl;
        }
        if (message.contains("end_pose")) {
            std::cout << "[UDP][" << direction << "] end_pose="
                      << message.at("end_pose").dump() << std::endl;
        }
        if (message.contains("joint_deg")) {
            std::cout << "[UDP][" << direction << "] joint_deg="
                      << message.at("joint_deg").dump() << std::endl;
        }
        std::cout << "[UDP][" << direction << "] "
                  << BuildCompactLogJson(message).dump() << std::endl;
        return;
    }

    nlohmann::json compact = message;
    compact.erase("command_id");
    std::cout << "[UDP][" << direction << "] " << compact.dump() << std::endl;
}

static void SendJsonToLastSender(UdpServer& server, const nlohmann::json& message) {
    const std::string payload = message.dump();
    LogUdpJson("TX", message);
    server.SendToLastSender(payload);
}

static void PrintPose(const std::string& label, const Pose6& pose) {
    std::cout << "[Command] " << label << "=[";
    for (std::size_t i = 0; i < pose.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << std::fixed << std::setprecision(6) << pose[i];
    }
    std::cout << "]" << std::endl;
}

static void PrintJoint(const std::string& label, const Joint6& joints_deg) {
    std::cout << "[Executor] " << label << "=[";
    for (std::size_t i = 0; i < joints_deg.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << std::fixed << std::setprecision(3) << joints_deg[i];
    }
    std::cout << "] deg" << std::endl;
}

static void PrintGraspCommand(const GraspCommand& command) {
    std::cout << "[Command] seq=" << command.seq
              << ", frame=" << command.frame
              << ", unit=" << command.unit
              << ", score=" << std::fixed << std::setprecision(6) << command.score
              << ", width=" << command.width << std::endl;
    PrintPose("pre_grasp_pose", command.pre_grasp_pose);
    PrintPose("grasp_pose", command.grasp_pose);
    PrintPose("lift_pose", command.lift_pose);
}

static bool ConfirmExecution() {
    std::cout << "[Executor] Execute this grasp? [y/N]: ";
    std::string answer;
    if (!std::getline(std::cin, answer)) {
        return false;
    }
    return answer == "y" || answer == "Y";
}

int main() {
    const std::string robot_ip = "192.168.1.20";
    const int robot_port = 8080;
    const int udp_port = 6556;

    RobotDriver robot;
    if (!robot.Connect(robot_ip, robot_port)) {
        std::cerr << "[Executor] robot connect failed" << std::endl;
        return 1;
    }
    std::cout << "[Executor] robot connected" << std::endl;

    PrintJoint("startup_home_joint", kDefaultHomeJointDeg);
    std::cout << "[Executor] moving to startup home" << std::endl;
    if (!robot.MoveJ(kDefaultHomeJointDeg, kDefaultMoveSpeed)) {
        std::cerr << "[Executor] failed to move startup home" << std::endl;
        return 1;
    }
    std::cout << "[Executor] startup home reached" << std::endl;
    std::cout << "[Executor] waiting for UDP commands on port "
              << udp_port << std::endl;

    SafetyConfig safety_config;
    GraspExecutor executor(robot, SafetyChecker(safety_config));
    UdpServer server(udp_port);
    if (!server.Start()) {
        return 1;
    }

    while (true) {
        std::string payload;
        if (!server.Receive(payload)) {
            continue;
        }

        nlohmann::json message;
        try {
            message = nlohmann::json::parse(payload);
        } catch (const std::exception& exc) {
            std::cerr << "[Executor] invalid JSON: " << exc.what() << std::endl;
            continue;
        }
        LogUdpJson("RX", message);

        const std::string type = message.value("type", "");
        const int seq = message.value("seq", 0);

        if (type == "pose_request") {
            std::cout << "[Executor] pose_request seq=" << seq
                      << ", reading robot state" << std::endl;
            RobotState state;
            if (!robot.CurrentState(state)) {
                nlohmann::json response = {
                    {"version", 1},
                    {"type", "pose_response"},
                    {"seq", seq},
                    {"status", "failed"},
                    {"reason", "failed to read robot state"},
                };
                SendJsonToLastSender(server, response);
                continue;
            }
            SendJsonToLastSender(server, MakePoseResponse(seq, state));
            continue;
        }

        if (type == "grasp_execute") {
            std::cout << "[Executor] grasp_execute seq=" << seq << std::endl;
            GraspCommand command;
            std::string error;
            if (!ParseGraspCommand(message, command, error)) {
                SendJsonToLastSender(server, MakeAck(seq, "", "rejected", error));
                continue;
            }
            PrintGraspCommand(command);

            if (!executor.Validate(command, error)) {
                SendJsonToLastSender(server, MakeAck(command.seq, command.command_id, "rejected", error));
                continue;
            }
            SendJsonToLastSender(server, MakeAck(command.seq, command.command_id, "accepted"));
            if (!ConfirmExecution()) {
                SendJsonToLastSender(
                    server,
                    MakeResult(command.seq, command.command_id, "cancelled", "operator declined")
                );
                continue;
            }
            if (executor.Execute(command, error)) {
                SendJsonToLastSender(server, MakeResult(command.seq, command.command_id, "success"));
            } else {
                SendJsonToLastSender(server, MakeResult(command.seq, command.command_id, "failed", error));
            }
            continue;
        }

        std::cerr << "[Executor] unknown message type: " << type << std::endl;
    }
}
