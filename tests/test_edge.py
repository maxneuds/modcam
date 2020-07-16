#!python
import cv2
import time
import pyfakewebcam
import numpy as np
import argparse
from threading import Thread


# args
def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('ndev', type=int, nargs='?', default=0)
  args = parser.parse_args()
  return(args.ndev)


# create caputre class
class Camcap:
  def __init__(self, src=0):
    # initialize the video camera stream and read the first frame
    # from the stream
    self.stream = cv2.VideoCapture(src)
    self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    self.stream.set(cv2.CAP_PROP_FPS, 30)
    (self.grabbed, self.frame) = self.stream.read()
    # initialize the variable used to indicate if the thread should
    # be stopped
    self.stopped = False

  # define threads
  def start(self):
    # start the thread to read frames from the video stream
    Thread(target=self.update, args=()).start()
    return self

  def update(self):
    # keep looping infinitely until the thread is stopped
    while True:
      # if the thread indicator variable is set, stop the thread
      if self.stopped:
        return
      # otherwise, read the next frame from the stream
      (self.grabbed, self.frame) = self.stream.read()

  def read(self):
    # return the frame most recently read
    return self.frame

  def stop(self):
    # indicate that the thread should be stopped
    self.stopped = True


# init cam capture
# setup access to the *real* webcam
ndev = get_args()
cam = f'/dev/video{ndev}'
cap = Camcap(src=cam).start()
height, width = 720, 960
# define margins
m_left = int(0.25 * width)
m_right = int(0.25 * width)
m_top = int(0.1 * height)
m_bot = int(0.4 * height)
crop_width = width - m_left - m_right
crop_height = height - m_top - m_bot
# setup the fake camera
fake_w = 1920
fake_h = 1080
# fake = pyfakewebcam.FakeWebcam('/dev/video20', width, height)
fake = pyfakewebcam.FakeWebcam('/dev/video20', fake_w, fake_h)

# define locations
file_name = 'cam2.jpg'

# define mods
blur_value = (41, 41)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg = cv2.createBackgroundSubtractorKNN()


def create_streamframe(im):
  height = fake_h
  width = fake_w
  # transform to frame height
  s = height / crop_height
  im_w = int(s * crop_width)
  dim = (im_w, height)
  im = cv2.resize(im, dim)
  m = int((width - im_w) / 2)
  frame = np.zeros((height, width, 3), np.uint8)
  frame[:, m:m + im_w] = im
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  return(frame)


def im_save(im, file_name):
  cv2.imwrite(file_name, frame)


while True:
  t_start = time.time()

  # read cam
  vcap = cap.read()
  # performance test with black screen
  # vcap = np.zeros((height, width, 3), np.uint8)

  # image processing
  im = vcap
  roi = im[m_top:-m_bot, m_left:-m_right]
  im_hflip = cv2.flip(roi, 1)
  frame = roi.copy()
  edges = cv2.Canny(frame, 100, 140)
  kernel = np.ones((3, 3), np.uint8)
  closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

  # show image
  try:
    cv2.imshow("cam", closing)
  except cv2.error:
    continue
  key = cv2.waitKey(1)

  # save capture to file
  # im_save(im, file_name)

  # stream capture to hook
  # fake webcam expects RGB
  # frame = create_streamframe(frame)
  # fake.schedule_frame(frame)

  if key == ord('q'):
    break
  t_end = time.time()
  t = t_end - t_start
  fps = np.around(1 / t, 2)
  # print(f'FPS: {fps}')

# shutdown cam
cv2.destroyAllWindows()
cap.stop()
