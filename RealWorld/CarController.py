import pickle
import queue
import socket
import struct  ## new
import threading
from math import sin, radians, degrees

import cv2
import pygame
from pygame.math import Vector2

import check_intersections
import getKeys
import helper
from helper import *

pygame.init()

src = "rtsp://pythonadmin:Pycamera@192.168.1.37:554/cam/realmonitor?channel=1&subtype=0"

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv2.aruco.DetectorParameters_create()

MAKE_CONN = True

mouseX = 0
mouseY = 0
using = "OUTER"
track_file = "track.npy"
LOAD_TRACK = True
font = pygame.font.SysFont("Calibri", 20)

pt1 = None
pt2 = None

car_width = 50
car_height = 100
CAR = pygame.image.load("car.png")
CAR = pygame.transform.smoothscale(CAR, (car_width, car_height))

if LOAD_TRACK:
    lines_outer, lines_inner = np.load(track_file, allow_pickle=True)
    lines = lines_outer
    lines.extend(lines_inner)
    using = "DONE"
else:
    lines_outer = []
    lines_inner = []

if MAKE_CONN:
    HOST = '192.168.1.2'
    PORT = 8123
    imageSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    imageSocket.bind((HOST, PORT))
    imageSocket.listen(10)
    conn, addr = imageSocket.accept()
    print('Connected by', addr)
    payload_size = struct.calcsize(">L")


def getData():
    if MAKE_CONN:
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
    else:
        return cv2.imread('frame.jpg')


prevAngle = 0


def sendData(speed, angle):
    global prevAngle
    prevAngle = angle
    speed = int(speed)
    angle = int(angle)
    if MAKE_CONN:
        conn.send(bytes(f"{speed} {angle}\n", encoding='utf-8'))
    # else:
    #     print(f"{speed} {angle}")


def findCar(image):
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
                                                       parameters=arucoParams)
    centers = []
    car = []
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # calculate angle
            angle = math.degrees(math.atan2(topRight[1] - topLeft[1], topRight[0] - topLeft[0]))
            #  if bottomRight[0] < bottomLeft[0] and bottomRight[1] > topRight[1]:
            #      angle = 360 - angle

            centers.append(((cX, cY), markerID, angle))
            if markerID == 822:
                car = ((cX, cY), angle)

            # convert each of the (x, y)-coordinate pairs to integers

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            # cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            # cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            # cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            ## compute and draw the center (x, y)-coordinates of the ArUco
            ## marker
            # cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            ## draw the ArUco marker ID on the image
            # cv2.putText(image, str(markerID),
            #            (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
            #            0.5, (0, 255, 0), 2)
    return car


class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


def takeAction(frame):
    sendData(1, 1)


camera = VideoCapture(src)

PREV_STEER = 0


class Car:
    def __init__(self, x, y, angle, length, max_steering=30, max_acceleration=5.0):
        self.position = Vector2(x, y)
        self.velocity = Vector2(0.0, 0.0)
        self.angle = angle
        self.length = length
        self.max_acceleration = max_acceleration
        self.max_steering = max_steering
        self.brake_deceleration = 10
        self.free_deceleration = 2
        self.max_velocity = 2000

    def move(self, speed, angle, dt=45):
        speed /= 200
        self.velocity = Vector2(speed, 0)
        self.velocity.x = max(-self.max_velocity, min(self.velocity.x, self.max_velocity))

        if angle:
            turning_radius = self.length / sin(radians(angle))
            angular_velocity = self.velocity.x / turning_radius
        else:
            angular_velocity = 0

        self.position += self.velocity.rotate(-self.angle) * dt
        self.angle += degrees(angular_velocity) * dt

    def getPosition(self):
        return [self.position.x, self.position.y]

    def create_sensors(self):
        self.sensors = list()
        self.create_sensor_dict()
        # print(sensor_dict)
        for i in self.sensor_dict:
            sensor = self.sensor_dict[i]
            self.sensors.append(self.create_sensor(sensor[0], sensor[1], 2000))

    def create_sensor(self, start_point, angle, line_length):
        angle = -self.angle + angle + 90
        x1 = start_point[0]
        y1 = start_point[1]
        endy = y1 + line_length * math.sin(math.radians(angle))
        endx = x1 + line_length * math.cos(math.radians(angle))
        end_point = (endx, endy)
        sensor = [(start_point[0], start_point[1]), end_point]
        return sensor

    def create_sensor_dict(self):
        self.sensor_dict = dict()

        # start_point = calculate_mid(self.car_position)
        self.sensor_dict["front"] = [self.getPosition(), -90]

        # start_point = calculate_mid(self.car["bottom_line"])
        self.sensor_dict["back"] = [self.getPosition(), 90]

        # start_point = calculate_corner(self.car_position, "right")
        # self.sensor_dict["right_forward"] = [self.getPosition(), -85]

        # start_point = calculate_corner(self.car_position, "left")
        # self.sensor_dict["left_forward"] = [self.getPosition(), -95]

        # start_point = calculate_corner(self.car_position, "right")
        self.sensor_dict["right_corner_top"] = [self.getPosition(), -45]

        # start_point = calculate_corner(self.car_position, "left")
        self.sensor_dict["left_corner_top"] = [self.getPosition(), -135]

        # start_point = calculate_corner(self.car_position, "right")
        self.sensor_dict["right_corner_side"] = [self.getPosition(), 0]

        # start_point = calculate_corner(self.car_position, "left")
        self.sensor_dict["left_corner_side"] = [self.getPosition(), -180]

        # start_point = calculate_corner(self.car_position, "right")
        self.sensor_dict["right_corner_bottom"] = [self.getPosition(), 135]

        # start_point = calculate_corner(self.car_position, "left")
        self.sensor_dict["left_corner_bottom"] = [self.getPosition(), 45]

    def calculate_sensors(self):
        self.create_sensors()
        self.sensor_dist = []
        for line in self.sensors:
            start_point = line[0]

            dists = [640.0]
            for lane in lines:
                track_lane = lane
                is_intersect = check_intersections.check_intersection(line, track_lane)
                if is_intersect:
                    point_of_intersect = check_intersections.calculate_intersection(line,
                                                                                    track_lane)  # first check if intersects
                    dist = helper.calculate_dist(start_point, point_of_intersect)
                    dists.append(dist)
            self.sensor_dist.append(min(dists))


class Simulate:
    def __init__(self, car_pos, car_angle):
        self.car = Car(car_pos[0], car_pos[1], car_angle, length=85)
        width = 1920
        height = 1080
        self.display = pygame.display.set_mode((width, height))
        self.step(0, 0)

    def draw_sensors(self):
        for i in range(len(self.car.sensors)):
            line = self.car.sensors[i]
            dist = self.car.sensor_dist[i]
            keys = list(self.car.sensor_dict.keys())

            angle = self.car.sensor_dict[keys[i]][1] - self.car.angle + 90

            endy = line[0][1] + dist * math.sin(math.radians(angle))
            endx = line[0][0] + dist * math.cos(math.radians(angle))
            end_point = (int(endx), int(endy))
            pygame.draw.line(self.display, (255, 0, 0), line[0], end_point, 1)
            pygame.draw.circle(self.display, (0, 0, 255), end_point, 5)

            sensorInfo = f"S: {i} D: {round(dist)}"
            text = font.render(sensorInfo, True, (255, 255, 255))
            self.display.blit(text, (end_point[0] + 10, end_point[1]))

    def getBestAction(self):
        global PREV_STEER
        speed = 40
        steer = 0
        maxSteer = 30
        # sensor_dist = [front, back, right_front, left_front, right_side, left_side, right_back, left_back]
        #                 0       1       2           3           4           5           6           7
        minLeftDist = 80
        minDelta = 9999999
        minAngle = 0
        choices = np.linspace(-maxSteer, maxSteer, num=20)
        for i in choices:
            nextCar = Car(self.car.position.x, self.car.position.y, self.car.angle, self.car.length)
            for _ in range(30):
                nextCar.move(speed, i, dt=10)
            nextCar.calculate_sensors()
            nextDelta = nextCar.sensor_dist[5] - minLeftDist
            nextDelta = abs(nextDelta)
            if nextDelta < minDelta:
                minDelta = nextDelta
                minAngle = i

        if self.car.sensor_dist[0] < 50:
            speed = 0
            print("STUCK")
        return speed, minAngle

    def simulateUser(self):
        keys = getKeys.key_check()
        speed = 0
        steer = 0
        maxSpeed = 30
        maxSteer = 30
        if "W" in keys:
            speed = maxSpeed
        elif "S" in keys:
            speed = -maxSpeed
        if "A" in keys:
            steer = maxSteer
        elif "D" in keys:
            steer = -maxSteer
        self.step(speed, steer)
        # frame = camera.read()
        # cv2.imshow("frame", frame)
        # getData()
        # sendData(speed, steer)

    def test(self):
        while True:
            # self.simulateUser()
            speed, steer = self.getBestAction()
            self.step(speed, steer)
            time.sleep(0.05)

    def step(self, speed, angle):
        self.move(speed, angle)
        self.car.calculate_sensors()
        self.render()

    def move(self, speed, angle):
        self.car.move(speed, angle)

    def draw_car(self):
        car = pygame.transform.rotate(CAR, self.car.angle - 90)
        car_rect = car.get_rect()
        car_rect.center = self.car.getPosition()
        self.display.blit(car, car_rect)

    def render(self):
        self.display.fill((0, 0, 0))
        pygame.draw.circle(self.display, (0, 0, 255), self.car.getPosition(), 5)
        for line in lines_outer:
            pygame.draw.line(self.display, (0, 255, 0), line[0], line[1], 2)
        for line in lines_inner:
            pygame.draw.line(self.display, (0, 255, 0), line[0], line[1], 2)
        self.draw_sensors()
        self.draw_car()
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()


class CarTrack:
    def __init__(self):
        self.car_position = [0, 0]
        self.car_angle = 90
        self.actionsLeft = 3
        self.maxActions = 1

    def getCarPosition(self):
        car = findCar(self.image)
        if not car or len(car) == 0:
            self.actionsLeft -= 1
        else:
            self.actionsLeft = self.maxActions
            self.car_position = car[0]
            self.car_angle = car[1]
            self.simulation = Simulate(self.car_position, -self.car_angle + 90)

    def getImage(self):
        image = camera.read()
        self.image = image

    def step(self):
        st = time.time()
        self.getImage()
        self.getCarPosition()
        self.render()
        speed = 20
        if self.actionsLeft <= 0:
            print("No car found")
            sendData(0, prevAngle)
        else:
            try:
                sendData(*self.getBestAction())
            except Exception as e:
                print(e)
                sendData(0, prevAngle)
        print("Getting")
        carView = getData()
        print("Got")
        print(f"FPS: {round(1 / (time.time() - st), 2)}")

    def getBestAction(self):
        # simulation.test()
        return self.simulation.getBestAction()

    def render(self):
        cv2.circle(self.image, self.car_position, 5, (0, 0, 255), -1)
        for line in lines_outer:
            cv2.line(self.image, line[0], line[1], (0, 255, 0), 2)
        for line in lines_inner:
            cv2.line(self.image, line[0], line[1], (0, 255, 0), 2)
        cv2.imshow('frame', self.image)
        cv2.waitKey(1)


def main():
    carTrack = CarTrack()
    cv2.namedWindow("frame")
    while True:
        carTrack.step()
        # cv2.setMouseCallback('frame', mouse_click)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print(e)
        conn.close()
        imageSocket.close()
