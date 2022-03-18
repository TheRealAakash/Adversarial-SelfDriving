import pickle
import socket
import struct  ## new

import cv2

from getKeys import key_check

HOST = '192.168.1.2'
PORT = 8123
imageSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
imageSocket.bind((HOST, PORT))
imageSocket.listen(10)
conn, addr = imageSocket.accept()
print('Connected by', addr)
payload_size = struct.calcsize(">L")


def getData():
    data = b""
    while len(data) < payload_size:
        data += conn.recv(4096)
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    # data = data[msg_size:]

    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    return frame


def sendData(speed, angle):
    conn.send(bytes(f"{speed} {angle}\n", encoding='utf-8'))


FRAME_IMAGE = "img_1.png"


def main():
    global FRAME_IMAGE
    frames = []
    while True:
        speed = 0
        angle = 0
        keys = key_check()
        maxAngle = 46
        maxSpeed = 60
        if "W" in keys:
            speed = maxSpeed
        if "A" in keys:
            angle = -maxAngle
        if "D" in keys:
            angle = maxAngle
        if "S" in keys:
            speed = -maxSpeed
        frame = getData()
        frames.append(frame)
        sendData(speed, angle)
        cv2.imshow('frame', frame)

        if "K" in keys:
            FRAME_IMAGE = "StopSign.png"
        if "L" in keys:
            FRAME_IMAGE = "img_1.png"
        if FRAME_IMAGE:
            img = cv2.imread(FRAME_IMAGE)
            img = cv2.resize(img, (700, 1120))
            cv2.imshow('frame2', img)
        if "Q" in keys:
            break
        cv2.waitKey(1)
    # save frames in a video
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, (frame.shape[1], frame.shape[0]))
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
    cv2.destroyAllWindows()


try:
    main()
except KeyboardInterrupt:
    conn.close()
    imageSocket.close()
    print("Connection closed")
