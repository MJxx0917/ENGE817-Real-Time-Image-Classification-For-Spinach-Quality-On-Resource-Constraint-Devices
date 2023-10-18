import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import Input
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def lr_schedule(epoch):
    return 0.01 * np.exp(-epoch / 10)

class_names = ["not very fresh", "not fresh", "fresh", "very fresh"]

data_dir = 'E:/dataset/Spinach_Quality'
image_dir = os.path.join(data_dir, 'image')
labels_df = pd.read_csv(os.path.join(data_dir, 'label_all.csv'), header=None, names=['label'])
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 20  

label_encoder = LabelEncoder()
labels_df['numerical_label'] = label_encoder.fit_transform(labels_df['label']).astype(str)  # Convert to string

labels_df['filename'] = labels_df['label'].apply(lambda x: os.path.join(image_dir, f"{int(x):04d}.jpg"))

labels_df = labels_df.sample(frac=1).reset_index(drop=True)

train_df, test_df = train_test_split(labels_df, test_size=0.4, stratify=labels_df['numerical_label'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_df['filename'] = train_df['filename'].astype(str)
test_df['filename'] = test_df['filename'].astype(str)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='numerical_label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='numerical_label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)  
x = Dropout(0.8)(x)  
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

lr_scheduler = LearningRateScheduler(lr_schedule)

class_weights = {}
for i, count in enumerate(train_df['numerical_label'].value_counts()):
    class_weights[i] = max(train_df.shape[0] / (len(class_names) * count), 1)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[lr_scheduler],
    class_weight=class_weights,  
    steps_per_epoch=len(train_generator),  
    validation_steps=len(test_generator)  
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")