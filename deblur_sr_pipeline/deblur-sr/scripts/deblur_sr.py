import numpy as np
from PIL import Image
import cv2
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Input
from keras.applications import VGG19
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from scipy.misc import imread, imresize
from math import log10, sqrt 
from deblurgan.model import generator_model
from deblurgan.utils import load_image, deprocess_image, preprocess_image
from keras.models import load_model
import keras.backend as K
import tensorflow as tf


def deblur(weight_path, input_dir, output_dir):
    g = generator_model()
    g.load_weights(weight_path)
    inputim = cv2.imread("./input_images/input.png")
    if len(inputim.shape) > 2 and inputim.shape[2] == 4:
       inputim = cv2.cvtColor(inputim, cv2.COLOR_BGRA2BGR)
    cv2.imwrite("./input_images/input.png", inputim)
    for image_name in os.listdir(input_dir):
        image = np.array([preprocess_image(load_image(os.path.join(input_dir, image_name)))])
        x_test = image
        generated_images = g.predict(x=x_test)
        generated = np.array([deprocess_image(img) for img in generated_images])
        x_test = deprocess_image(x_test)
        for i in range(generated_images.shape[0]):
            x = x_test[i, :, :, :]
            img = generated[i, :, :, :]
            original = cv2.imread("./original/original.png")
            if len(original.shape) > 2 and original.shape[2] == 4:
                original = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
            original = imresize(original, (512,512,3))
            img1 = imresize(img, (512,512,3))
            original = Image.fromarray(original.astype(np.uint8))
            original.save("./original/original.png")
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img1 = Image.fromarray(img1.astype(np.uint8))
            img1.save("./output_deblur/deblur_generated.png")

def super_resolution(data_dir):
    epochs = 3000
    batch_size = 1
    mode = 'predict'
    loss1 = []
    loss2 = []
    psnrv = []

    # Shape of low-resolution and high-resolution images
    low_resolution_shape = (96, 96, 3)
    high_resolution_shape = (384, 384, 3)

    # Common optimizer for all networks
    common_optimizer = Adam(0.0002, 0.5)
    if mode == 'predict':

        # Build the generator network
        generator = build_generator()

        # Load models
        #generator.load_weights("generator_sr.h5")
        loss = VGG_LOSS(image_shape) 
        generator = load_model("./gen_model3000.h5", custom_objects={'vgg_loss': loss.vgg_loss})
        
        #discriminator.load_weights("./discriminator_sr.h5")

        # Get 10 random images
        high_resolution_images, low_resolution_images = sample_images(data_dir=data_dir, batch_size=1,
                                                                      low_resolution_shape=low_resolution_shape,
                                                                      high_resolution_shape=high_resolution_shape)
        
        # Normalize images
        #high_resolution_images = high_resolution_images / 127.5 - 1.
        low_resolution_images = low_resolution_images / 127.5 - 1.

        # Generate high-resolution images from low-resolution images
        generated_images = generator.predict_on_batch(low_resolution_images)
        temp = 0
        c = 0
        original = cv2.imread("./original/original.png")
        if len(original.shape) > 2 and original.shape[2] == 4:
           original = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
        deblur_output = cv2.imread("./output_deblur/deblur_generated.png")
        if len(deblur_output.shape) > 2 and deblur_output.shape[2] == 4:
           deblur_output = cv2.cvtColor(deblur_output, cv2.COLOR_BGRA2BGR)
        for index, img in enumerate(generated_images):
          img = imresize(img, (512,512,3))
          save_images(deblur_output, original, img,path="./results/img_{}".format(index))

image_shape = (384,384,3)
class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
    
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
    
        return K.mean(K.square(model(y_true) - model(y_pred)))

def residual_block(x):
    """
    Residual block
    """
    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = "same"
    momentum = 0.8
    activation = "relu"

    res = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding)(x)
    res = Activation(activation=activation)(res)
    res = BatchNormalization(momentum=momentum)(res)

    res = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(res)
    res = BatchNormalization(momentum=momentum)(res)

    # Add res and x
    res = Add()([res, x])
    return res

def build_generator():
    """
    Create a generator network using the hyperparameter values defined below
    :return:
    """
    residual_blocks = 16
    momentum = 0.8
    input_shape = (64, 64, 3)

    # Input Layer of the generator network
    input_layer = Input(shape=input_shape)

    # Add the pre-residual block
    gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)

    # Add 16 residual blocks
    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)

    # Add the post-residual block
    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)

    # Take the sum of the output from the pre-residual block(gen1) and the post-residual block(gen2)
    gen3 = Add()([gen2, gen1])

    # Add an upsampling block
    gen4 = UpSampling2D(size=2)(gen3)
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
    gen4 = Activation('relu')(gen4)

    # Add another upsampling block
    gen5 = UpSampling2D(size=2)(gen4)
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
    gen5 = Activation('relu')(gen5)

    # Output convolution layer
    gen6 = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen5)
    output = Activation('tanh')(gen6)

    # Keras model
    model = Model(inputs=[input_layer], outputs=[output], name='generator')
    return model


def sample_images(data_dir, batch_size, high_resolution_shape, low_resolution_shape):
    # Make a list of all images inside the data directory
    all_images = glob.glob(data_dir)

    # Choose a random batch of images
    images_batch = np.random.choice(all_images, size=batch_size)

    low_resolution_images = []
    high_resolution_images = []

    for img in images_batch:
        # Get an ndarray of the current image
        img1 = imread(img, mode='RGB')
        img1 = img1.astype(np.float32)

        # Resize the image
        img1_high_resolution = imresize(img1, high_resolution_shape)
        img1_low_resolution = imresize(img1, low_resolution_shape)

        # Do a random horizontal flip
        """if np.random.random() < 0.5:
            img1_high_resolution = np.fliplr(img1_high_resolution)
            img1_low_resolution = np.fliplr(img1_low_resolution)"""

        high_resolution_images.append(img1_high_resolution)
        low_resolution_images.append(img1_low_resolution)

    # Convert the lists to Numpy NDArrays
    return np.array(high_resolution_images), np.array(low_resolution_images)


def save_images(low_resolution_image, original_image, generated_image, path):
    low_resolution_image=Image.fromarray(low_resolution_image.astype(np.uint8))
    original_image=Image.fromarray(original_image.astype(np.uint8))
    generated_image = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)
    generated_image=Image.fromarray(generated_image.astype(np.uint8))

    low_resolution_image.save("./results/deblur_op.png")
    original_image.save("./results/original.png")
    generated_image.save("./results/deblur_sr_op.png")


def compute():
    weight_path = "./generator.h5"
    input_dir = "input_images"
    output_dir = "/"
    deblur(weight_path, input_dir, output_dir)
    data_dir = "./output_deblur/*.png"
    super_resolution(data_dir)

compute()