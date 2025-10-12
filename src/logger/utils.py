import io

import matplotlib.pyplot as plt
import PIL
from torchvision.transforms import ToTensor

plt.switch_backend("agg")


def plot_images(imgs, config):
    names = config.writer.names

    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        img = imgs[i].permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(names[i])
        axes[i].axis("off")

    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    image = ToTensor()(PIL.Image.open(buf))
    plt.close()
    return image


def plot_spectrogram(spectrogram, name=None):
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.pcolormesh(spectrogram)
    if name:
        ax.set_title(name)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)

    buf.seek(0)
    img = PIL.Image.open(buf).convert("RGB")
    img.load()
    buf.close()
    return img
