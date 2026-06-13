#include "udp_server.h"

#include <iostream>

UdpServer::UdpServer(int port) : port_(port) {}

UdpServer::~UdpServer() {
    Stop();
}

bool UdpServer::Start() {
#ifndef _WIN32
    std::cerr << "[UdpServer] Windows Winsock implementation is required." << std::endl;
    return false;
#else
    WSADATA wsadata;
    if (WSAStartup(MAKEWORD(2, 2), &wsadata) != 0) {
        std::cerr << "[UdpServer] WSAStartup failed" << std::endl;
        return false;
    }
    wsa_initialized_ = true;

    socket_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_ == INVALID_SOCKET) {
        std::cerr << "[UdpServer] socket create failed" << std::endl;
        Stop();
        return false;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(static_cast<u_short>(port_));
    if (bind(socket_, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) == SOCKET_ERROR) {
        std::cerr << "[UdpServer] bind failed on port " << port_ << std::endl;
        Stop();
        return false;
    }
    std::cout << "[UdpServer] listening on 0.0.0.0:" << port_ << std::endl;
    return true;
#endif
}

void UdpServer::Stop() {
#ifdef _WIN32
    if (socket_ != INVALID_SOCKET) {
        closesocket(socket_);
        socket_ = INVALID_SOCKET;
    }
    if (wsa_initialized_) {
        WSACleanup();
        wsa_initialized_ = false;
    }
#endif
}

bool UdpServer::Receive(std::string& message) {
#ifndef _WIN32
    (void)message;
    return false;
#else
    char buffer[8192];
    int sender_len = sizeof(last_sender_);
    int recv_len = recvfrom(
        socket_,
        buffer,
        static_cast<int>(sizeof(buffer) - 1),
        0,
        reinterpret_cast<sockaddr*>(&last_sender_),
        &sender_len
    );
    if (recv_len <= 0) {
        return false;
    }
    buffer[recv_len] = '\0';
    message.assign(buffer, static_cast<std::size_t>(recv_len));
    return true;
#endif
}

bool UdpServer::SendToLastSender(const std::string& message) {
#ifndef _WIN32
    (void)message;
    return false;
#else
    int result = sendto(
        socket_,
        message.c_str(),
        static_cast<int>(message.size()),
        0,
        reinterpret_cast<sockaddr*>(&last_sender_),
        sizeof(last_sender_)
    );
    return result != SOCKET_ERROR;
#endif
}

