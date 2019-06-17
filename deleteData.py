import csv
import cv2
import numpy as np
import os
import threading
import time

useMyData = True


lines = []
startlocation = "data/data"
if useMyData == True:
    startlocation = "training"

csvPath = startlocation+'/driving_log.csv'

with open(csvPath) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# images = []
# measurements = []
# for line in lines:
#     cwd = os.getcwd()
#     sourcepath = line[0]
#     filename = sourcepath.split('/')[-1]
#     current_path = cwd + "/" + startlocation+ '/IMG/' + filename
#     image = cv2.imread(current_path)
#     if (type(image) is np.ndarray):
#         images.append(image)
#         measurement = float(line[3])
#         measurements.append(measurement)

def deleteEntry(lineNumber):
    print("Delete entry - "+lines[lineNumber][0])
    del lines[lineNumber]

def saveCSV():
    writer = csv.writer(open(csvPath, 'w'))
    for row in lines:
        writer.writerow(row)

def getMeasurement(lineNumber):
    line = lines[lineNumber]
    measurement = float(line[3])
    return measurement

def getImageNumber(lineNumber):
    cwd = os.getcwd()
    line = lines[lineNumber]
    sourcepath = line[0]
    filename = sourcepath.split('/')[-1]
    current_path = cwd + "/" + startlocation+ '/IMG/' + filename
    image = cv2.imread(current_path)
    return image


currentFrame = 0
released = True

class ThreadingExample(object):

    def __init__(self, interval=1):
        """ Constructor
        :type interval: int
        :param interval: Check interval, in seconds
        """
        self.interval = interval

        thread = threading.Thread(target=self.run, args=())
        thread.daemon = True                            # Daemonize thread
        thread.start()                                  # Start the execution

    def rotateImage(self, img, angle):
        num_rows, num_cols = img.shape[:2]

        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
        return img_rotation

    def run(self):
        """ Method that runs forever """
        while True:
            global currentFrame

            temp = getImageNumber(currentFrame)
            angle = getMeasurement(currentFrame) * -60
            height, width, depth = temp.shape
            newimg = cv2.resize(temp, (width * 3, height * 3))
            newimg = cv2.cvtColor(newimg, cv2.COLOR_RGB2RGBA)

            s_img = cv2.imread("up.png", -1)
            s_img = self.rotateImage(s_img, angle)
            s_img = cv2.resize(s_img, (50,50))
            y_offset = 400
            x_offset = 50
            y1, y2 = y_offset, y_offset + s_img.shape[0]
            x1, x2 = x_offset, x_offset + s_img.shape[1]

            alpha_s = s_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                newimg[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                          alpha_l * newimg[y1:y2, x1:x2, c])

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(newimg, str(currentFrame), (10, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('image', newimg)
            cv2.waitKey(1)


example = ThreadingExample()

from pynput.keyboard import Key, Listener
from pynput import keyboard

def on_press(key):
    global currentFrame, released
    print('{0} pressed'.format(key))
    if key == Key.down and released == True:
        currentFrame = currentFrame - 1
        if currentFrame < 0:
            currentFrame = 0
        return
        # released = False
    if key == Key.up and released == True:
        currentFrame = currentFrame + 1
        if currentFrame >= len(lines):
            currentFrame = len(lines)-1
        return
        # released = False
    if key == Key.backspace and released == True:
        deleteEntry(currentFrame)
        if currentFrame >= len(lines):
            currentFrame = len(lines)-1
        return
    if key == Key.space:
        currentFrame = currentFrame + 50
        if currentFrame >= len(lines):
            currentFrame = len(lines)-1
        return
    if key.char == 's':
        print("save file")
        saveCSV()
    print(currentFrame)


def on_release(key):
    global released
    released = True
    # print('{0} release'.format(key))
    if key == Key.esc:
        # Stop listener
        return False

# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()
