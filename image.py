import cv2
import argparse
from yolo import Detector

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='The path to the image you want to run a detection on.')

args = ap.parse_args()

image = cv2.imread(args.image)

detector = Detector()

detector.detect_objects(image, draw=True)

cv2.imshow('Detections', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
