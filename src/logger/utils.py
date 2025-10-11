import io

import matplotlib.pyplot as plt
import PIL
from torchvision.transforms import ToTensor

plt.switch_backend("agg")  # fix RuntimeError: main thread is not in main loop


def plot_images(imgs, config):
    """
    Combine several images into one figure.

    Args:
        imgs (Tensor): array of images (B X C x H x W).
        config (DictConfig): hydra experiment config.
    Returns:
        image (Tensor): a single figure with imgs plotted side-to-side.
    """
    # name of each img in the array
    names = config.writer.names
    # figure size
    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        # channels must be in the last dim
        img = imgs[i].permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(names[i])
        axes[i].axis("off")  # we do not need axis
    # To create a tensor from matplotlib,
    # we need a buffer to save the figure
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    # convert buffer to Tensor
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
