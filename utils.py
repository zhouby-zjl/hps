import socket

class utils:
    @staticmethod
    def getIpIn32bits(ip_str):
        return socket.inet_aton(ip_str)

    @staticmethod
    def getIpInStr(ip_32bits):
        return socket.inet_ntoa(ip_32bits)
