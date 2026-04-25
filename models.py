import tensorflow as tf
from tensorflow.keras import layers

LATENT = 5

# ================= AE =================
def build_autoencoder():

    inp = tf.keras.Input(shape=(64,64,1))

    x = layers.Conv2D(32,3,2,"same")(inp)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64,3,2,"same")(x)
    x = layers.ReLU()(x)

    x = layers.Flatten()(x)
    z = layers.Dense(LATENT)(x)

    encoder = tf.keras.Model(inp, z)

    z_in = tf.keras.Input((LATENT,))
    x = layers.Dense(16*16*64)(z_in)
    x = layers.Reshape((16,16,64))(x)

    x = layers.Conv2DTranspose(64,3,2,"same",activation="relu")(x)
    x = layers.Conv2DTranspose(32,3,2,"same",activation="relu")(x)

    out = layers.Conv2D(1,3,padding="same",activation="sigmoid")(x)

    decoder = tf.keras.Model(z_in, out)

    model = tf.keras.Model(inp, decoder(encoder(inp)))
    model.compile(optimizer="adam", loss="mse")

    return model


# ================= VAE =================
def build_vae():

    inp = tf.keras.Input((64,64,1))

    x = layers.Conv2D(32,3,2,"same",activation="relu")(inp)
    x = layers.Conv2D(64,3,2,"same",activation="relu")(x)
    x = layers.Flatten()(x)

    mu = layers.Dense(LATENT)(x)
    log_var = layers.Dense(LATENT)(x)

    def sample(m, v):
        eps = tf.random.normal(tf.shape(m))
        return m + tf.exp(0.5*v)*eps

    z = layers.Lambda(lambda t: sample(t[0], t[1]))([mu, log_var])

    encoder = tf.keras.Model(inp, [mu, log_var, z])

    z_in = tf.keras.Input((LATENT,))
    x = layers.Dense(16*16*64, activation="relu")(z_in)
    x = layers.Reshape((16,16,64))(x)

    x = layers.Conv2DTranspose(64,3,2,"same",activation="relu")(x)
    x = layers.Conv2DTranspose(32,3,2,"same",activation="relu")(x)

    out = layers.Conv2D(1,3,padding="same",activation="sigmoid")(x)

    decoder = tf.keras.Model(z_in, out)

    class VAE(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.enc = encoder
            self.dec = decoder

        def call(self, x):
            _, _, z = self.enc(x)
            return self.dec(z)

        def train_step(self, data):
            x,_ = data

            with tf.GradientTape() as tape:
                m,v,z = self.enc(x)
                recon = self.dec(z)

                recon_loss = tf.reduce_mean(tf.square(x-recon))
                kl = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1+v-tf.square(m)-tf.exp(v), axis=1)
                )

                loss = recon_loss + 20.0 * kl

            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            return {"loss": loss}

    model = VAE()
    model.compile(optimizer="adam")

    return model
