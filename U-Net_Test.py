import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
import numpy as np
from tqdm import tqdm
import cv2
import random

Image_W = 128  # This determine the dimensions for the input layer in...(Width)
Image_H = 128  # This determine the dimensions for the input layer in...(Height)
Image_C = 3  # This determine the dimensions for the input layer in...(Color Channels, '3' for RBG)

TRAIN_PATH = 'data-science-bowl-2018/stage1_train/'  # This is where I have the downloaded training images (change to where you save them)
TEST_PATH = 'data-science-bowl-2018/stage1_test/'  # This is where I have the downloaded testing images (change to where you save them)

training_ids = next(os.walk(TRAIN_PATH))[
    1]  # Once accessed this lists the training ids (the file names that match the image names)
testing_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((len(training_ids), Image_W, Image_W, Image_C), dtype=np.uint8)  # This creates a uint8 numpy array filled with zeros that match the target dimensions of the input layer and the volume of the training data
Y_train = np.zeros((len(training_ids), Image_W, Image_W, 1), dtype=np.bool)

# Note:
# The Following section was designed by Screenivas (https://github.com/bnsreenu) to tackle the issue of
# the mask images being separated by cell instead of total cells. This combines them in a way the testing
# data should have been for processing'
###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###
print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(training_ids), total=len(training_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :Image_C]
    img = resize(img, (Image_H, Image_W), mode='constant', preserve_range=True)
    X_train[n] = img  # Fill Empty X_Train with value from image
    mask = np.zeros((Image_H, Image_W, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (Image_H, Image_W), mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# X_Test (testing images)
X_test = np.zeros((len(testing_ids), Image_H, Image_W, Image_C), dtype=np.uint8)
sizes_test = []
print('Resizing Test Images')

for n, id_ in tqdm(enumerate(testing_ids), total=len(testing_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :Image_C]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (Image_H, Image_W), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')
###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###_###


# Here we initiate a random sampling of X_train and Y_Train
image_x = random.randint(0, len(training_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()
# If it looks good we are clear to move on with building a model


# Build the model
# Note:
# This was done with the guidance of Screenivas (https://github.com/bnsreenu) via his Youtube tutorial series

inputs = tf.keras.layers.Input((Image_W, Image_H, Image_C))
# The inputs have to be floating point values not ints between 1 and 255
# So we have to divide them all by 255

# "Contracting Path" of the U-Net
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# "Expansion" Path of the U-Net
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#####
# ADD a Model Checkpoint
#####

checkpointer = tf.keras.callbacks.ModelCheckpoint('Model_for_nuclei.h5', verbose=1, save_best_only=True)
# This is pretty useful, it saves a copy of the model and goes for the best version

# Now we need early stopping
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    # Very useful, it stops the epochs when validation loss is consistent "patience" amount of times
    tf.keras.callbacks.TensorBoard(log_dir='logs')
]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=100, callbacks=callbacks)

############## VALIDATION_TIME#######################

preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)  # These set up thresholds for the probability of each component
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

print("Done!")