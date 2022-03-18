from RealWorld.raspi.picar import front_wheels, back_wheels
import cv2

fw = front_wheels.Front_Wheels(debug=False)
bw = back_wheels.Back_Wheels(debug=False)
SPEED = 1
fw.turn_straight()
cam = cv2.VideoCapture(0)
while True:
    ret, image = cam.read()
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
