# dependencies for transformations
from PIL import Image
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
import pandas

# dependencies for extended transformations
from numpy import expand_dims
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import ImageDataGenerator

# Load original image data from the npy file
data = np.load(r'C:\Users\berni\documents\Elephant Seal Research Project\noses\image_data.npy', allow_pickle=True)
bb_data = np.load(r'C:\Users\berni\documents\Elephant Seal Research Project\noses\bb_data_small.npy', allow_pickle=True)

# -----------------------------------------
# Image transformations
final_images = []
final_bb = []
def translate_left_right():
    for image in data:
        # Flip image from left to right
        flipped_img = tf.image.flip_left_right(image)
        final_images.append(flipped_img)
    for image in bb_data:
        flipped_bb = tf.image.flip_left_right(image)
        final_bb.append(flipped_bb)

def flip_top_bottom():
    for image in data:
        # Flip image from top to bottom
        ver_img = data.transpose(Image.FLIP_TOP_BOTTOM)
        final_images.append(ver_img)
    for image in bb_data:
        ver_bb = data.transpose(Image.FLIP_TOP_BOTTOM)
        final_bb.append(ver_bb)

def rotate():
    for image in data:
        # Rotate 180
        rotated_img = tf.image.rot90(tf.image.rot90(image))
        final_images.append(rotated_img)
    for image in bb_data:
        rotated_bb = tf.image.rot90(tf.image.rot90(image))
        final_bb.append(rotated_bb)

# -------------- Extended Tranformations -----------------

# Random Brightness Augmentation

image_array = img_to_array(data) # convert to numpy array
samples = expand_dims(data, 0) # expand dimension to one sample
datagen = ImageDataGenerator(brightness_range=[0.2,1.0]) # create image data augmentation generator
it = datagen.flow(samples, batch_size=1) # prepare iterator

def brightness():
    for i in data:
        pyplot.subplot(330 + 1 + i) # define subplot
        batch = it.next() # generate batch of images
        image = batch[0].astype('uint8') # convert to unsigned integers for viewing
        # pyplot.imshow(image) # plot raw pixel data
        # pyplot.show() # show the figure
        final_images.append(image)

# ------------------------------------

# Random Rotations

def rotations():
    for i in data:
        pyplot.subplot(330 + 1 + i) # define subplot
        batch = it.next() # generate batch of images
        image = batch[0].astype('uint8') # convert to unsigned integers for viewing
        # pyplot.imshow(image) # plot raw pixel data
        # pyplot.show() # show the figure
        final_images.append(image)

# -----------------------------------

# Converts all images in a directory to '.npy' format. Use np.save and np.load to save and load the images

# Path to image directory
# path = "/noses/transformed_images/"
# dirs = os.listdir(path)
# dirs.sort()
# x_train=[]

# def load_dataset():
#     # Append images to a list
#     for item in dirs:
#         if os.path.isfile(path+item):
#             im = Image.open(path+item).convert("RGB")
#             im = np.array(im)
#             x_train.append(im)

# if __name__ == "__main__":
    
#     load_dataset()
    
# Convert and save the list of images in '.npy' format
img_set=np.array(final_images)
bb_set=np.array(final_bb)
np.save("transformed_images.npy", img_set)
np.save("transformed_bb.npy", bb_set)