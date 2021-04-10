import os 
import torch
import numpy as np
import matplotlib.pyplot as plt 

def plot_loss(train_loss_list, valid_loss_list, title):
    plt.plot(train_loss_list, label='Training')
    plt.plot(valid_loss_list, label='Validation')
    plt.xlabel('Epoch')
    plt.title(title)
    plt.legend(frameon=False)


def plot_images(images, labels, classes, normalize = False):
    n_images = len(images)
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (10, 10))

    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        image = images[i]

        if normalize:
            image_min = image.min()
            image_max = image.max()
            image.clamp_(min = image_min, max = image_max)
            image.add_(-image_min).div_(image_max - image_min + 1e-5)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(classes[labels[i]])
        ax.axis('off')
