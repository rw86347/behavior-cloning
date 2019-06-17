import cv2
import csv
import numpy as np
import os
import sklearn

def LoadCSVData(dataPath, correction): #load images and
    dir = [x[0] for x in os.walk(dataPath)]
    dataDir = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), dir))
    center = []
    left = []
    right = []
    steering = []
    directory = dataDir[0]

    lines = []
    with open(dataPath + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            lines.append(line)

    center = []
    left = []
    right = []
    measurements = []
    for line in lines:
        if line[0] == 'center':
            continue
        measurements.append(float(line[3]))
        center.append(directory + '/' + line[0].strip())
        left.append(directory + '/' + line[1].strip())
        right.append(directory + '/' + line[2].strip())
    center.extend(center)
    left.extend(left)
    right.extend(right)
    steering.extend(measurements)

    images = []
    images.extend(center)
    images.extend(left)
    images.extend(right)
    measurements = []
    measurements.extend(steering)
    measurements.extend([x + correction for x in steering])
    measurements.extend([x - correction for x in steering])
    return (images, measurements)

def generator(data, batch_size=32):
    num_samples = len(data)
    while 1:
        data = sklearn.utils.shuffle(data)
        for offset in range(0, num_samples, batch_size):
            batch_samples = data[offset:offset + batch_size]
            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                originalImage = cv2.imread(imagePath)
                if (type(originalImage) is np.ndarray):
                    image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    angles.append(measurement)
                    # Flip image
                    images.append(cv2.flip(image,1))
                    angles.append(measurement*-1.0)

            # trim image
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

def neuralNet():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


imagePaths, measurements = LoadCSVData('data', 0.2)
print('Total Images: {}'.format( len(imagePaths)))
from sklearn.model_selection import train_test_split
data = list(zip(imagePaths, measurements))
train_samples, validation_samples = train_test_split(data, test_size=0.2)
print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))
g_train = generator(train_samples, batch_size=32)
print("g_train")
print(g_train)
g_valid = generator(validation_samples, batch_size=32)
print("g_valid")
print(g_valid)

model = neuralNet()
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(g_train, samples_per_epoch=len(train_samples), validation_data=g_valid, nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

model.save('model.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()