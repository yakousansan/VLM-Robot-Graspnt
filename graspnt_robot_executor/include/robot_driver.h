#ifndef GRASPNT_ROBOT_EXECUTOR_ROBOT_DRIVER_H
#define GRASPNT_ROBOT_EXECUTOR_ROBOT_DRIVER_H

#include <array>
#include <string>

#include "protocol.h"
#include "rm_service.h"

class RobotDriver {
public:
    RobotDriver();
    ~RobotDriver();

    bool Connect(const std::string& ip, int port);
    void Disconnect();
    bool ConfigureModbus();
    bool OpenGripper();
    bool CloseGripper();
    bool CurrentState(RobotState& state);
    bool MoveJ(const Joint6& joints_deg, double speed);
    bool MoveL(const Pose6& pose, double speed);
    bool TargetReachable(const Pose6& pose);

private:
    bool WriteModbusRegister(int address, int data);
    bool ReadModbusRegister(int address, int& out_data);
    bool ReadJointDegree(std::array<float, 6>& joints_deg);
    bool FillCurrentEnd2Base(Matrix4& h_end2base);

    static constexpr int kGripperTargetPosReg = 0x0103;
    static constexpr int kGripperCurrentStateReg = 0x0201;
    static constexpr int kGripperCurrentPosReg = 0x0202;
    static constexpr int kGripperOpenPos = 1000;
    static constexpr int kGripperClosePos = 0;

    bool connected_ = false;
    RM_Service robotic_arm_;
    rm_robot_handle* robot_handle_ = nullptr;
};

#endif
