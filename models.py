import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, Conv2DTranspose, LeakyReLU, Dropout, UpSampling2D, BatchNormalization
from matplotlib import pyplot as plt
import numpy as np
import math

models = tf.keras.models
layers = tf.keras.layers

def build_generator():
    model = models.Sequential()

    model.add(layers.Dense(8 * 8 * 512, input_dim=128))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((8, 8, 512)))    
    
    model.add(layers.Conv2DTranspose(256, 4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(128, 4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(64, 4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    
    # model.add(UpSampling2D())
    # model.add(Conv2D(256, kernel_size=5, padding='same'))
    # model.add(LeakyReLU(alpha=0.2))

    # model.add(layers.Conv2D(256, kernel_size=4, padding='same'))
    # model.add(layers.LeakyReLU(alpha=0.2))
    # model.add(layers.BatchNormalization())

    # # Final output layer
    # model.add(layers.Conv2D(3, kernel_size=7, padding='same', activation='tanh', dtype='float32'))
    
    model.add(layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="tanh"))
    
    return model

def build_discriminator():
    model = models.Sequential()
    
    model.add(layers.Conv2D(16, kernel_size=5, input_shape=(128, 128, 3)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))
    
    model.add(layers.Conv2D(32, kernel_size=5))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(64, kernel_size=5))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))

    model.add(layers.Conv2D(128, kernel_size=5))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.4))

    # model.add(layers.Conv2D(256, kernel_size=5))
    # model.add(layers.LeakyReLU(alpha=0.2))
    # model.add(layers.Dropout(0.4))

    model.add(layers.Flatten())
    # model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1))
    
    return model

def build_critic(img_shape=(128, 128, 3)):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(
        64, 4, strides=2, padding="same",
        input_shape=img_shape
    ))
    model.add(tf.keras.layers.LeakyReLU(0.2))

    model.add(tf.keras.layers.Conv2D(
        128, 4, strides=2, padding="same"
    ))
    model.add(tf.keras.layers.LeakyReLU(0.2))

    model.add(tf.keras.layers.Conv2D(
        256, 4, strides=2, padding="same"
    ))
    model.add(tf.keras.layers.LeakyReLU(0.2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))  # NO sigmoid

    return model



# G = build_generator()
# G.summary()
# D = build_discriminator()
# D.summary()
# img = G.predict(np.random.normal(0, 1, (4, 128)), verbose=0)
# # print("Generated image shape:", img.shape)
# C = build_critic()
# C.summary()

# imgs = (img + 1) / 2  # Rescale to [0, 1]

# # plt.figure(figsize=(6,6))
# # for i in range(4):
# #     plt.subplot(2,2,i+1)
# #     plt.imshow(imgs[i])
# #     plt.axis("off")
# #     plt.imsave(f"generated_image_{i}.png", imgs[i])

# # plt.tight_layout()
# # plt.close()

# print("Image generation completed.")

# print(D.predict(img))