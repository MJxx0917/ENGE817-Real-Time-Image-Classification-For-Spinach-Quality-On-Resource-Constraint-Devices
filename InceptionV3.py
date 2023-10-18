import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    return 0.001 * np.exp(-epoch / 10)

class_names = ["not very fresh", "not fresh", "fresh", "very fresh"]

data_dir = 'E:/dataset/Spinach_Quality'
image_dir = os.path.join(data_dir, 'image')
labels_df = pd.read_csv(os.path.join(data_dir, 'label_all.csv'), header=None, names=['label'])
IMG_SIZE = (299, 299)
BATCH_SIZE = 32
EPOCHS = 20

label_encoder = LabelEncoder()
labels_df['numerical_label'] = label_encoder.fit_transform(labels_df['label']).astype(str) 

labels_df['filename'] = labels_df['label'].apply(lambda x: os.path.join(image_dir, f"{int(x):04d}.jpg"))

# Split the data into train and test sets
split_index = int(0.8 * len(labels_df))
train_df = labels_df[:split_index].copy()
test_df = labels_df[split_index:].copy()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
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

print("Train Generator Info:")
print(train_generator.filepaths[:10])
print("Test Generator Info:")
print(test_generator.filepaths[:10])

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

lr_scheduler = LearningRateScheduler(lr_schedule)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[lr_scheduler]
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

model.save('spinach_quality_model.h5')
