#ifndef GRASPNT_ROBOT_EXECUTOR_UDP_SERVER_H
#define GRASPNT_ROBOT_EXECUTOR_UDP_SERVER_H

#include <string>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

class UdpServer {
public:
    explicit UdpServer(int port);
    ~UdpServer();

    bool Start();
    void Stop();
    bool Receive(std::string& message);
    bool SendToLastSender(const std::string& message);

private:
    int port_ = 6556;

#ifdef _WIN32
    bool wsa_initialized_ = false;
    SOCKET socket_ = INVALID_SOCKET;
    sockaddr_in last_sender_{};
#endif
};

#endif

