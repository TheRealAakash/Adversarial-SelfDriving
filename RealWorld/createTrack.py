import pickle
import socket
import struct  ## new
import time

import cv2

import helper, check_intersections
from helper import *
from imutils.video import VideoStream

src = "rtsp://pythonadmin:Pycamera@192.168.1.37:554/cam/realmonitor?channel=1&subtype=0"
camera = cv2.VideoCapture(src)
MAKE_CONN = False

mouseX = 0
mouseY = 0
using = "OUTER"
track_file = "track.npy"
LOAD_TRACK = False

pt1 = None
pt2 = None

lines_outer = []
lines_inner = []
lines = []


def mouse_click(event, x, y,
                flags, param):
    global mouseX, mouseY, pt1, using, pt2, lines
    # to check if left mouse
    # button was clicked
    mouseX = x
    mouseY = y
    if event == cv2.EVENT_LBUTTONDOWN:
        if using == "OUTER":
            if not pt1:
                pt1 = (x, y)
            elif len(lines_outer) == 0:
                lines_outer.append((pt1, (x, y)))
            else:
                # calculate distance between 2 points
                dist = math.sqrt(
                    (pt1[0] - x) ** 2 + (pt1[1] - y) ** 2)
                if dist < 10:
                    lines_outer.append((lines_outer[-1][1], pt1))
                    using = "INNER"
                    pt1 = None
                    print("USING INNER")
                else:
                    lines_outer.append((lines_outer[-1][1], (x, y)))
        elif using == "INNER":
            if not pt2:
                pt2 = (x, y)
            elif len(lines_inner) == 0:
                lines_inner.append((pt2, (x, y)))
            else:
                # calculate distance between 2 points
                dist = math.sqrt(
                    (pt2[0] - x) ** 2 + (pt2[1] - y) ** 2)
                if dist < 10:
                    lines_inner.append((lines_inner[-1][1], pt2))
                    lines = lines_inner
                    lines.extend(lines_outer)
                    np.save(track_file, [lines_inner, lines_outer])
                    using = "DONE"
                    print("DONE")
                else:
                    lines_inner.append((lines_inner[-1][1], (x, y)))


def main():
    cv2.namedWindow("frame")
    cv2.setMouseCallback('frame', mouse_click)
    while True:
        ret, frame = camera.read()
        if ret:
            for line in lines_outer:
                cv2.line(frame, line[0], line[1], (0, 255, 0), 2)
            for line in lines_inner:
                cv2.line(frame, line[0], line[1], (0, 255, 0), 2)
            if using == "DONE":
                break
            cv2.imshow("frame", frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()
