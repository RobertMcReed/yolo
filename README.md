# YOLO

Perform object detection on your computer or RaspberryPi.

## Installation

Grab a copy of the yolov3 weights using wget:

```
wget https://pjreddie.com/media/files/yolov3.weights
```

If on a Pi see the note below, otherwise create a new [virtualenv](https://docs.python-guide.org/dev/virtualenvs/) and install the dependencies:

```
pip install -r requirements.txt
```



### RaspberryPi

Before running `pip install -r requirements.txt` make sure you have the Pi-specific dependencies installed.

```
sudo apt-get install libhdf5-dev libhdf5-serial-dev
sudo apt-get install libqtwebkit4 libqt4-test
```

If you are using the PiCamera you must also install the dependency:

```
pip install "picamera[array]"
```

## Usage

For both image and video, to exit the script press "q" while the image viewer is active.

### Image

Run object detection on an image:

```
python image.py --i img_test.jpg
```

### Video

Run object detection from a webcam:

#### Built-in Camera

```
python video.py
```

#### USB Webcam

```
python video.py --src 1
```

#### Pi Camera

```
python video.py --pi
```

#### Threading

Process objects in a separate thread, resulting in better fps, but laggy detections.

```
python video.py --threaded
```
