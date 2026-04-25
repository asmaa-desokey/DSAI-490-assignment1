import matplotlib.pyplot as plt

def plot_loss(history, title):
    plt.plot(history.history['loss'])
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
