from sympy import Q
import zmq
import time
from IK import IK_LM

def send_array(socket, A):
    msg = dict(
        dtype = str(A.dtype),
        shape = A.shape,
        data = A.tolist()
    )
    socket.send_json(msg)

def start_server():
    while True:
        cmd = input("cmd - (server/quit): ")
        if cmd == "server":
            print("Server On")
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            socket.bind("tcp://*:5555")
            while True:
                # wait for receive message
                msg = socket.recv()
                time.sleep(0.5)
                if msg.decode("utf-8") == "Start Compute":   
                    # compute and send as json dict
                    print("Start compute")
                    a = IK_LM().LM()
                    send_array(socket, a)
                elif msg.decode("utf-8") == "Shutdown":
                    print("Client disconnect")
                    break
        elif cmd == "quit":
            print("bye")
            break

if __name__ == "__main__":
    start_server()