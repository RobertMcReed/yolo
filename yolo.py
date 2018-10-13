import numpy as np
import cv2
from multiprocessing import Process, Queue

# constants
weights = 'yolov3.weights'
classesPath = 'yolov3.txt'
config = 'yolov3.cfg'
scale = 0.00392
conf_thresh = 0.5
nms_thresh = 0.4


class Detector:
    def __init__(self, threaded=False):
        self.set_classes_and_colors()
        self.read_net()
        self.threaded = threaded

        if threaded:
            self.inputQueue = Queue(maxsize=1)
            self.outputQueue = Queue(maxsize=1)

            p = Process(target=self.detect_with_thread)
            p.daemon = True
            p.start()
            self.last_detections = None

    def set_classes_and_colors(self):
        # read the class names from yolov3.txt
        classes = None
        with open(classesPath, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        # each class will have a different color
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        self.classes = classes
        self.colors = colors

    def read_net(self):
        # read the pretrained model and configs
        net = cv2.dnn.readNet(weights, config)
        self.net = net

    def prep_input_and_get_outs(self, image):
        # create the input blob
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True,
                                     crop=False)

        # set the input for the neural net
        self.net.setInput(blob)

        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in
                         self.net.getUnconnectedOutLayers()]

        # gather the predictions from the output layers
        outs = self.net.forward(output_layers)

        return outs

    def draw_bounding_box(self, img, class_id, confidence, p1, p2):
        """Draw bounding boxes on the detected object with its class name"""

        label = str(self.classes[class_id])
        label = '{}: {:.2f}'.format(label, confidence)

        color = self.colors[class_id]

        cv2.rectangle(img, p1, p2, color, 2)

        x, y = p1[:2]

        if y - 10 > 0:
            Y = y - 10
            X = x
        else:
            Y = y + 10
            X = x + 10

        cv2.putText(img, label, (X, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

    def detect_objects(self, image, draw=False):
        height, width = image.shape[:2]

        # gather the predictions from the output layers
        outs = self.prep_input_and_get_outs(image)

        # initializations
        class_ids = []
        confs = []
        boxes = []

        # for each detection in each output layer,
        # get the confidence, class_id, bounding box params,
        # ignoring weak predictions
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confs.append(float(conf))
                    boxes.append([x, y, w, h])

        # apply non-maxima suppression
        indices = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, nms_thresh)

        detections = {'class_ids': class_ids,
                      'confs': confs,
                      'boxes': boxes,
                      'indices': indices}

        if draw:
            self.draw_detections(image=image, **detections)

        return detections

    def handle_image(self, image):
        if self.threaded:
            detections = None

            if self.inputQueue.empty():
                self.inputQueue.put(image)

            if not self.outputQueue.empty():
                detections = self.outputQueue.get()

            if detections is not None:
                self.draw_detections(image=image, **detections)
                self.last_detections = detections
            elif self.last_detections is not None:
                self.draw_detections(image, **self.last_detections)
        else:
            self.detect_objects(image, draw=True)

    def draw_detections(self, image, indices, boxes, confs, class_ids):
        # iterate through the remaining detections and draw the bounding box
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box[:4]

            self.draw_bounding_box(image, class_ids[i], confs[i],
                                   (round(x), round(y)),
                                   (round(x + w), round(y + h)))

    def detect_with_thread(self):
        while True:
            # check to see if there is a frame in the input queue
            if not self.inputQueue.empty():
                # grab the frame, run the detection, and add it to the queue
                image = self.inputQueue.get()

                detections = self.detect_objects(image)
                self.outputQueue.put(detections)
