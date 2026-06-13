#include "robot_driver.h"

#include <array>
#include <cmath>
#include <iostream>

static std::array<float, 6> ToFloatArray(const Joint6& values) {
    std::array<float, 6> result{};
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i] = static_cast<float>(values[i]);
    }
    return result;
}

static rm_pose_t ToRmPose(const Pose6& pose) {
    rm_pose_t target_pose{};
    target_pose.position.x = static_cast<float>(pose[0]);
    target_pose.position.y = static_cast<float>(pose[1]);
    target_pose.position.z = static_cast<float>(pose[2]);
    target_pose.euler.rx = static_cast<float>(pose[3]);
    target_pose.euler.ry = static_cast<float>(pose[4]);
    target_pose.euler.rz = static_cast<float>(pose[5]);
    return target_pose;
}

static Matrix4 PoseToMatrix(const rm_pose_t& pose) {
    const double tx = pose.position.x;
    const double ty = pose.position.y;
    const double tz = pose.position.z;

    const double qw = pose.quaternion.w;
    const double qx = pose.quaternion.x;
    const double qy = pose.quaternion.y;
    const double qz = pose.quaternion.z;

    Matrix4 matrix{{
        {{
            1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
            2.0 * qx * qy - 2.0 * qz * qw,
            2.0 * qx * qz + 2.0 * qy * qw,
            tx,
        }},
        {{
            2.0 * qx * qy + 2.0 * qz * qw,
            1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
            2.0 * qy * qz - 2.0 * qx * qw,
            ty,
        }},
        {{
            2.0 * qx * qz - 2.0 * qy * qw,
            2.0 * qy * qz + 2.0 * qx * qw,
            1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
            tz,
        }},
        {{0.0, 0.0, 0.0, 1.0}},
    }};
    return matrix;
}

RobotDriver::RobotDriver() = default;

RobotDriver::~RobotDriver() {
    Disconnect();
}

bool RobotDriver::Connect(const std::string& ip, int port) {
    if (connected_) {
        std::cout << "[RobotDriver] connection already exists" << std::endl;
        return true;
    }
    robotic_arm_.rm_init(RM_TRIPLE_MODE_E);
    robot_handle_ = robotic_arm_.rm_create_robot_arm(ip.c_str(), port);
    if (robot_handle_ == nullptr) {
        std::cerr << "[RobotDriver] arm connect failed: null handle" << std::endl;
        return false;
    }
    if (robot_handle_->id == -1) {
        std::cerr << "[RobotDriver] arm connect failed: invalid id" << std::endl;
        robotic_arm_.rm_delete_robot_arm(robot_handle_);
        robot_handle_ = nullptr;
        return false;
    }

    connected_ = true;
    std::cout << "[RobotDriver] connected " << ip << ":" << port << std::endl;
    return true;
}

void RobotDriver::Disconnect() {
    if (!connected_) {
        return;
    }
    if (robot_handle_ != nullptr) {
        robotic_arm_.rm_close_modbus_mode(robot_handle_, 1);
        robotic_arm_.rm_delete_robot_arm(robot_handle_);
        robot_handle_ = nullptr;
    }
    connected_ = false;
    std::cout << "[RobotDriver] disconnect" << std::endl;
}

bool RobotDriver::ConfigureModbus() {
    if (!connected_) {
        return false;
    }
    if (robotic_arm_.rm_set_modbus_mode(robot_handle_, 1, 115200, 10) == 0) {
        return true;
    }
    std::cerr << "[RobotDriver] configure Modbus failed" << std::endl;
    return false;
}

bool RobotDriver::OpenGripper() {
    return WriteModbusRegister(kGripperTargetPosReg, kGripperOpenPos);
}

bool RobotDriver::CloseGripper() {
    return WriteModbusRegister(kGripperTargetPosReg, kGripperClosePos);
}

bool RobotDriver::CurrentState(RobotState& state) {
    if (!connected_) {
        return false;
    }
    rm_current_arm_state_t current_state{};
    if (robotic_arm_.rm_get_current_arm_state(robot_handle_, &current_state) != 0) {
        std::cerr << "[RobotDriver] failed to get current arm state" << std::endl;
        return false;
    }

    state.end_pose = {
        current_state.pose.position.x,
        current_state.pose.position.y,
        current_state.pose.position.z,
        current_state.pose.euler.rx,
        current_state.pose.euler.ry,
        current_state.pose.euler.rz,
    };

    std::array<float, 6> joints_deg{};
    if (ReadJointDegree(joints_deg)) {
        for (std::size_t i = 0; i < joints_deg.size(); ++i) {
            state.joint_deg[i] = joints_deg[i];
        }
    } else {
        for (std::size_t i = 0; i < state.joint_deg.size(); ++i) {
            state.joint_deg[i] = current_state.joint[i];
        }
    }

    if (!FillCurrentEnd2Base(state.h_end2base)) {
        state.h_end2base = PoseToMatrix(current_state.pose);
    }
    return true;
}

bool RobotDriver::MoveJ(const Joint6& joints_deg, double speed) {
    if (!connected_) {
        return false;
    }
    std::array<float, 6> joints = ToFloatArray(joints_deg);
    if (robotic_arm_.rm_movej(robot_handle_, joints.data(), static_cast<float>(speed), 0, 0, 1) == 0) {
        return true;
    }
    std::cerr << "[RobotDriver] movej failed" << std::endl;
    return false;
}

bool RobotDriver::MoveL(const Pose6& pose, double speed) {
    if (!connected_) {
        return false;
    }
    rm_pose_t target_pose = ToRmPose(pose);
    if (robotic_arm_.rm_movel(robot_handle_, target_pose, static_cast<float>(speed), 0, 0, 1) == 0) {
        return true;
    }
    std::cerr << "[RobotDriver] movel failed" << std::endl;
    return false;
}

bool RobotDriver::TargetReachable(const Pose6& pose) {
    if (!connected_) {
        return false;
    }
    rm_current_arm_state_t current_state{};
    if (robotic_arm_.rm_get_current_arm_state(robot_handle_, &current_state) != 0) {
        std::cerr << "[RobotDriver] failed to get current state for IK" << std::endl;
        return false;
    }

    rm_inverse_kinematics_params_t ik_params{};
    ik_params.q_pose = ToRmPose(pose);
    ik_params.flag = 1;
    for (int i = 0; i < 6; ++i) {
        ik_params.q_in[i] = current_state.joint[i];
    }

    float q_out[6] = {0.0f};
    int ik_result = rm_algo_inverse_kinematics(robot_handle_, ik_params, q_out);
    if (ik_result != 0) {
        std::cerr << "[RobotDriver] target unreachable, IK error code: " << ik_result << std::endl;
        return false;
    }
    return true;
}

bool RobotDriver::WriteModbusRegister(int address, int data) {
    if (!connected_ || robot_handle_ == nullptr) {
        return false;
    }

    rm_peripheral_read_write_params_t params{};
    params.port = 1;
    params.address = address;
    params.device = 1;
    params.num = 1;

    if (robotic_arm_.rm_write_single_register(robot_handle_, params, data) == 0) {
        return true;
    }
    std::cerr << "[RobotDriver] Modbus write failed, address=" << address << std::endl;
    return false;
}

bool RobotDriver::ReadModbusRegister(int address, int& out_data) {
    if (!connected_ || robot_handle_ == nullptr) {
        return false;
    }

    rm_peripheral_read_write_params_t params{};
    params.port = 1;
    params.address = address;
    params.device = 1;
    params.num = 1;

    if (robotic_arm_.rm_read_holding_registers(robot_handle_, params, &out_data) == 0) {
        return true;
    }
    std::cerr << "[RobotDriver] Modbus read failed, address=" << address << std::endl;
    return false;
}

bool RobotDriver::ReadJointDegree(std::array<float, 6>& joints_deg) {
    if (!connected_ || robot_handle_ == nullptr) {
        return false;
    }
    if (robotic_arm_.rm_get_joint_degree(robot_handle_, joints_deg.data()) == 0) {
        return true;
    }
    std::cerr << "[RobotDriver] failed to get joint degrees" << std::endl;
    return false;
}

bool RobotDriver::FillCurrentEnd2Base(Matrix4& h_end2base) {
    std::array<float, 6> joints_deg{};
    if (!ReadJointDegree(joints_deg)) {
        return false;
    }

    rm_pose_t pose = rm_algo_forward_kinematics(robot_handle_, joints_deg.data());
    h_end2base = PoseToMatrix(pose);
    return true;
}
