import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, LeakyReLu, Dropout, Upsampling2D, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
import math

models = tf.keras.models
layers = tf.keras.layers

def build_generator():
    
    model = models.Sequential()
    
    model.add(layers.Dense(16 * 16 * 256, input_dim=128))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((16, 16, 256)))
    
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(256, kernel_size=5, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization())
    
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(256, kernel_size=5, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization())
    
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(256, kernel_size=5, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization())
    
    
    model.add(layers.Conv2D(256, kernel_size=5, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(256, kernel_size=7, padding="same", activation="tanh", dtype="float32"))
    
    return model

def build_discriminator():
    model = models.Sequential()
    
    model.add(layers.Conv2D(16, kernal_size=5, input_size=(128, 128, 3)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))
    
    model.add(layers.Conv2D(32, kernel_size=5))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.4))
    
    model.add(layers.Conv2D(64, kernel_size=5))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.4))
    
    model.add(layers.Conv2D(128, kernel_size=5))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.4))
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))
    
    return model

G = build_generator()
G.summary()