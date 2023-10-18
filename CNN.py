import cv2
import os
import numpy as np
import pandas as pd
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score
import psutil
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.metrics import precision_score, recall_score

TRAIN_DIR = 'E:/dataset/Spinach_Quality/image'
IMG_SIZE = 50
LR = 1e-4
MODEL_NAME = 'spinach_quality-{}-{}.h5'.format(LR, 'enhanced')

label_mapping = {
    1: 'not_very_fresh',
    2: 'not_fresh',
    3: 'fresh',
    4: 'very_fresh'
}

def label_img(img, labels_df):
    img_id = int(img.split('.')[0]) - 1
    label = labels_df.iloc[img_id, 0]
    quality_label = label_mapping[label]
    label = [int(quality_label == qlabel) for qlabel in label_mapping.values()]
    return label

def create_spinach_data():
    images = []
    labels = []

    labels_df = pd.read_csv('E:/dataset/Spinach_Quality/label_all.csv', header=None)

    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img, labels_df)
        if label is not None:
            path = os.path.join(TRAIN_DIR, img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)

    combined_data = list(zip(images, labels))
    shuffle(combined_data)
    images[:], labels[:] = zip(*combined_data)

    return images, labels

train_images, train_labels = create_spinach_data()

X = np.array(train_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
Y = np.array(train_labels)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(4, activation='softmax'))

optimizer = Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()
history = model.fit(datagen.flow(X_train, Y_train, batch_size=32), epochs=20, validation_data=(X_test, Y_test))
training_time = time.time() - start_time

loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Testing Accuracy: {accuracy * 100:.2f}%')

