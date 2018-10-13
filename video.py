import cv2
from imutils.video import VideoStream, FPS
from multiprocessing import Process, Queue
from time import sleep
import os
from yolo import Detector
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-s', '--src', type=int, default=0,
                help='Defaults to built-in camera, use 1 for a usb camera.')
ap.add_argument('-p', '--pi', action="store_true", default=False,
                help='Whether to use the pi camera. Overrides src.')
ap.add_argument('-t', '--threaded', action="store_true", default=False,
                help='Whether to use threading or not.')

args = ap.parse_args()
threaded = args.threaded
thread_status = 'a separate' if args.threaded else 'the same'

detector = Detector(threaded)

print('Starting video stream...')
print('Running detector in {} thread...'.format(thread_status))
videoSettings = {'src': args.src} if not args.pi else {'usePiCamera': True}
vs = VideoStream(**videoSettings).start()

sleep(2)

fps = FPS().start()

while True:
    image = vs.read()

    detector.handle_image(image)

    fps.update()

    cv2.imshow('Detections', image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

fps.stop()
print('Elapsed time: {:.2f}'.format(fps.elapsed()))
print('Approx FPS: {:.2f}'.format(fps.fps()))
cv2.waitKey(1)
cv2.destroyAllWindows()
vs.stop()
sleep(0.5)
cv2.waitKey(1)
