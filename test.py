import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import os
import math

# from train_loop import GAN, GANMonitor, d_opt, g_opt, d_loss, g_loss
from train_loop_W import WGAN, GANMonitor, g_opt, c_opt
from models import build_generator, build_discriminator, build_critic
# from tensorflow.keras.optimizers import Adam

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, Conv2DTranspose, LeakyReLU, Dropout, UpSampling2D

# gpu = tf.config.experimental.list_physical_devices('GPU')
# if gpu:
#     try:
#         tf.config.experimental.set_memory_growth(gpu[0], True)
#         print("Memory growth for GPU set to True")
#     except RuntimeError as e:
#         print(e)
# else:
#     print("No GPU found")
    

# ===== CONFIG =====
IMG_SIZE = 128   
BATCH_SIZE = 8
DATASET_DIR = "train_anime"  # path to your dataset folder
checkpoint_dir = "./WGAN_checkpoints"
epoch_var = tf.Variable(0, trainable=False)

os.makedirs(checkpoint_dir, exist_ok=True)

# ===== IMAGE LOADER =====
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)

    # Normalize to [-1, 1] for GAN (tanh)
    img = (img / 127.5) - 1.0
    return img

# ===== DATASET PIPELINE =====
dataset = (
    tf.data.Dataset.list_files(DATASET_DIR + "/*.png", shuffle=True)
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    # .cache()
    .shuffle(1000)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

# print(dataset)

# for batch in dataset.take(1):
#     print("Shape:", batch.shape)
#     print("Range:", tf.reduce_min(batch).numpy(),
#                     tf.reduce_max(batch).numpy())

# ===== Data visualization =====


# for batch in dataset.take(1):
#     imgs = (batch + 1) / 2
#     n = imgs.shape[0]

#     cols = 8
#     rows = math.ceil(n / cols)

#     plt.figure(figsize=(cols * 2, rows * 2))
#     for i in range(n):
#         plt.subplot(rows, cols, i + 1)
#         plt.imshow(imgs[i])
#         plt.axis("off")

#     plt.savefig("preview.png")
#     plt.close()

# def build_generator():
#     model = Sequential()

#     model.add(Dense(16 * 16 * 256, input_dim=128))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Reshape((16, 16, 256)))  
    
#     model.add(UpSampling2D())
#     model.add(Conv2D(256, kernel_size=5, padding='same'))
#     model.add(LeakyReLU(alpha=0.2))
    
#     return model

# G = build_generator()
# G.summary()

G = build_generator()
# D = build_discriminator()
C = build_critic()

gan = WGAN(G, C)

gan.compile(g_opt, c_opt)

# checkpoint = tf.train.Checkpoint(
#     generator = gan.G,
#     discriminator = gan.D,
#     g_optimizer = gan.g_opt, 
#     d_optimizer = gan.d_opt,
#     epoch = epoch_var,
# )

# checkpoint_manager = tf.train.CheckpointManager(
#     checkpoint,
#     checkpoint_dir,
#     max_to_keep=5
# )

checkpoint = tf.train.Checkpoint(
    generator=G,
    critic=C,
    g_optimizer=g_opt,
    c_optimizer=c_opt
)

checkpoint_manager = tf.train.CheckpointManager(
    checkpoint,
    checkpoint_dir,
    max_to_keep=5
)


print("GAN compiled successfully!")

if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    print("Restored from", checkpoint_manager.latest_checkpoint)
else:
    print("No checkpoint found. Training from scratch.")

hist = gan.fit(
    dataset,
    epochs=300,
    callbacks=[GANMonitor(checkpoint_manager)]
)