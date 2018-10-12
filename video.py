import cv2
from imutils.video import VideoStream, FPS
from multiprocessing import Process, Queue
import time
import os
from yolo import Detector
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--src', type=int, default=0,
                help='Defaults to built-in camera, use 1 for a usb camera.')
ap.add_argument('-p', '--pi', type=bool, default=False,
                help='Whether to use the pi camera. Overrides src.')
ap.add_argument('-t', '--threaded', type=bool, default=False,
                help='Whether to use threading or not.')

args = ap.parse_args()

detector = Detector()

print('Starting video stream...')
videoSettings = {'src': args.src} if not args.pi else {'usePiCamera': True}
vs = VideoStream(**videoSettings).start()

time.sleep(2)

fps = FPS().start()

while True:
    image = vs.read()
    detector.detect_objects(image)
    fps.update()

    cv2.imshow('Detections', image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

fps.stop()
print('Elapsed time: {:.2f}'.format(fps.elapsed()))
print('Approx FPS: {:.2f}'.format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()
