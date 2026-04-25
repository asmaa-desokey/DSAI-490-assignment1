import os
from data_loader import build_dataset
from models import build_autoencoder, build_vae
from utils import plot_loss

DATA_PATH = "../data"

def train_all():

    classes = os.listdir(DATA_PATH)

    for cls in classes:
        path = os.path.join(DATA_PATH, cls)
        if not os.path.isdir(path):
            continue

        print("Training:", cls)

        ds = build_dataset(path)

        # AE
        ae = build_autoencoder()
        hist_ae = ae.fit(ds, epochs=5)
        plot_loss(hist_ae, f"AE Loss - {cls}")

        # VAE
        vae = build_vae()
        hist_vae = vae.fit(ds, epochs=5)
        plot_loss(hist_vae, f"VAE Loss - {cls}")


if __name__ == "__main__":
    train_all()
