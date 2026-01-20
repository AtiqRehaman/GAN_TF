# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import BinaryCrossentropy
from models import build_generator, build_discriminator
import tensorflow as tf
import os
from datetime import datetime
# from tensorflow.keras.models import Model  # Base model

timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
out_dir = f"./generated_images/{timestamp}"

os.makedirs(out_dir, exist_ok=True)

optimizer = tf.keras.optimizers

g_opt = optimizer.Adam(learning_rate=0.0001)
c_opt = optimizer.Adam(learning_rate=0.0001)

# g_loss = tf.keras.losses.BinaryCrossentropy()
# d_loss = tf.keras.losses.BinaryCrossentropy()

class WGAN(tf.keras.Model):
    def __init__(self, G, C, latent_dim=128, gp_weight=10.0):
        super().__init__()
        self.G = G
        self.C = C
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight

    def compile(self, g_opt, c_opt):
        super().compile()
        
        self.g_opt = g_opt
        self.c_opt = c_opt
        
    def critic_loss(self, real_score, fake_score):
        return tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)

    def generator_loss(self, fake_score):
        return -tf.reduce_mean(fake_score)

    def gradient_penalty(self, critic, real_images, fake_images):
        batch_size = tf.shape(real_images)[0]
        epsilon = tf.random.uniform([batch_size, 1, 1, 1])
        
        interpolated = epsilon * real_images + (1 - epsilon) * fake_images
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = critic(interpolated, training=True)
            
        grads = tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        
        return gp
        
    def train_step(self, real_images):  # W-GAN
        batch_size = tf.shape(real_images)[0]

        for _ in range(5):  # n_critic = 5
            noise = tf.random.normal((batch_size, self.latent_dim))

            with tf.GradientTape() as c_tape:
                fake_images = self.G(noise, training=True)

                real_scores = self.C(real_images, training=True)
                fake_scores = self.C(fake_images, training=True)

                c_loss = self.critic_loss(real_scores, fake_scores)
                gp = self.gradient_penalty(self.C, real_images, fake_images)
                total_c_loss = c_loss + self.gp_weight * gp

            c_grads = c_tape.gradient(
                total_c_loss, self.C.trainable_variables
            )
            self.c_opt.apply_gradients(
                zip(c_grads, self.C.trainable_variables)
            )

        noise = tf.random.normal((batch_size, self.latent_dim))

        with tf.GradientTape() as g_tape:
            fake_images = self.G(noise, training=True)
            fake_scores = self.C(fake_images, training=True)
            g_loss = self.generator_loss(fake_scores)

        g_grads = g_tape.gradient(
            g_loss, self.G.trainable_variables
        )
        self.g_opt.apply_gradients(
            zip(g_grads, self.G.trainable_variables)
        )

        return {
            "c_loss": c_loss,
            "g_loss": g_loss,
            "gp": gp
        }
    
# G = build_generator()
# D = build_discriminator()

# gan = GAN(G, D)
# gan.compile(g_opt, d_opt, g_loss, d_loss)

# Build callback

# from tensorflow.keras.preprocessing.image import array_to_img
# from tensorflow.keras.callbacks import Callback

class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager=None,num_img=1, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.checkpoint_manager = checkpoint_manager
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1)% 10 != 0:
            return
        
        # Generate noise
        random_latent_vectors = tf.random.normal(
            (self.num_img, self.latent_dim)
        )

        # Generate images (inference mode)
        generated_images = self.model.G(
            random_latent_vectors, training=False
        )

        # Convert from [-1, 1] → [0, 255]
        generated_images = (generated_images + 1.0) * 127.5
        generated_images = tf.cast(generated_images, tf.uint8)

        # Save images
        for i in range(self.num_img):
            img = tf.keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(
                os.path.join(
                    out_dir,
                    f"generated_img_{epoch:03d}_{i+1}.png"
                )
            )
            
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.save()
            # print(f"Checkpoint saved at epoch {epoch + 1}")