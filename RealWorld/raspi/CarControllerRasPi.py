import pickle
import time
import picar
from picar.SunFounder_PCA9685 import Servo
import cv2
import socket
import struct
import re

picar.setup()
PORT = 8123

MOTOR_A = 17
MOTOR_B = 27

PWM_A = 4
PWM_B = 5
BUS_NUMBER = 1

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


class Car:
    def __init__(self):
        self.speed = 1
        self.back_wheels = picar.back_wheels.Back_Wheels()
        self.front_wheels = picar.front_wheels.Front_Wheels()
        # self.cam = cv2.VideoCapture(0)
        self.pan_servo = Servo.Servo(1)
        self.tilt_servo = Servo.Servo(2)
        self.client_socket = None
        self.makeConnection()

    def makeConnection(self):
        connected = False
        while not connected:
            try:
                self.stopControl()
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # print("Connecting...")
                self.client_socket.connect(('192.168.1.2', PORT))
                print("Connected")
                connected = True
            except Exception as e:
                pass
                # print("Connection failed")

    def set_speed(self, speed):
        self.speed = speed
        self.back_wheels.speed = self.speed

    def setSteeringAngle(self, angle):
        self.front_wheels.wheel.write(angle)

    def forward(self, speed):
        self.set_speed(speed)
        self.back_wheels.forward()

    def getImage(self):
        ret, frame = self.cam.read()
        while not ret:
            ret, frame = self.cam.read()
        return frame

    def sendData(self):
        try:
            frame = self.getImage()
            result, frame = cv2.imencode('.jpg', frame, encode_param)
            data = pickle.dumps(frame, 0)
            size = len(data)
            self.client_socket.send(struct.pack(">L", size) + data)
        except Exception as e:
            print(e)
            self.makeConnection()

    def getControls(self):
        try:
            data = self.client_socket.recv(1024)
            data = data.decode('utf-8')
            data = data.replace('\n', '')
            data = data.split()
            speed = -int(data[0])
            angle = int(data[1]) + 90
            self.setSteeringAngle(angle)
            self.set_speed(abs(speed))
            if speed == 0:
                self.back_wheels.stop()
            if speed > 0:
                self.back_wheels.forward()
            if speed < 0:
                self.back_wheels.backward()
        except Exception as e:
            print(e)
            self.makeConnection()

    def stopControl(self):
        self.set_speed(0)
        self.back_wheels.stop()
        self.front_wheels.wheel.write(90)

    def close(self):
        self.stopControl()
        self.client_socket.close()


if __name__ == '__main__':
    car = Car()
    try:
        while True:
            car.sendData()
            car.getControls()
    except KeyboardInterrupt:
        pass
    car.close()
