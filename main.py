import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


DATA_DIR = 'C:/Users/ZBook/Desktop/PetImages'
CATEGORIES = ['Dog', 'Cat']
IMG_SIZE = 50
training_data = []


# for category in CATEGORIES:
#     path = os.path.join(DATA_DIR, category)
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # 1/3 size needed
#         plt.imshow(img_array, cmap='gray')
#         plt.show()
#         new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#         plt.imshow(new_array, cmap='gray')
#         plt.show()

#
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # 1/3 size needed
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()
print(len(training_data))

random.shuffle(training_data)

x = []  # features
y = []  # labels

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)
#
# pickle_out = open("x.pickle", "wb")
# pickle.dump(x, pickle_out)
# pickle_out.close()
#
# pickle_out = open("y.pickle", "wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()

# pickle_in = open("x.pickle", "rb")
# x = pickle.load(pickle_in)
#
# pickle_in = open("y.pickle", "rb")
# y = pickle.load(pickle_in)

x = x/255.0   # RGB-Values

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))  # 2D

model.add(Flatten())
model.add(Dense(64))  # 1D

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(x, y, batch_size=32, epochs=10, validation_split=0.1)
