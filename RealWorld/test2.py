import subprocess
import time

import cv2
import ffmpeg_streaming
import numpy as np
import queue
import threading

src = "rtsp://aakash:aaichaball1@192.168.1.37/cam/realmonitor?channel=1&subtype=0"
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams.adaptiveThreshConstant = 1
print(arucoParams.adaptiveThreshConstant)
print(dir(arucoParams)
      )

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
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


camera = VideoCapture(src)
last = time.time()
while True:
    # detect aruco
    img = camera.read()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)

    # draw aruco
    img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
    cv2.imshow("img", img)
    print("FPS: ", round(1 / (time.time() - last)))
    last = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
