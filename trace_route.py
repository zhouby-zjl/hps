import os
import sys
import socket

def trace_route(destination, max_hops=30):
    ttl = 1
    port = 33434
    while ttl <= max_hops:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sender.setsockopt(socket.SOL_IP, socket.IP_TTL, ttl)

        receiver.bind(("", port))
        sender.sendto(b"", (destination, port))

        try:
            data, addr = receiver.recvfrom(1024)
            print(f"{ttl}\t{addr[0]}")
            if addr[0] == destination:
                print("Trace complete.")
                break
        except socket.error:
            print(f"{ttl}\t*")
        finally:
            receiver.close()
            sender.close()
        ttl += 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <destination>")
        sys.exit(1)

    destination = sys.argv[1]
    trace_route(destination)