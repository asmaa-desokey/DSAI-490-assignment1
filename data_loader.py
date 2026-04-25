import tensorflow as tf
import os

IMG_SIZE = (64, 64)
BATCH = 32

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, IMG_SIZE)
    return img / 255.0

def build_dataset(folder):
    paths = [os.path.join(folder, f) for f in os.listdir(folder)]

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: (x, x))

    ds = ds.shuffle(500).batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

def add_noise(x):
    noise = tf.random.normal(tf.shape(x), stddev=0.3)
    return tf.clip_by_value(x + noise, 0, 1)
