import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os

output_path = 'comparison_post_processing_plot.pdf'

image_paths = ['./images/post processing difference/images/test_155.png', 
               './images/post processing difference/images/test_155.png', 
               './images/post processing difference/images/test_169.png', 
               './images/post processing difference/images/test_169.png', 
               './images/post processing difference/images/test_155.png',
               './images/post processing difference/images/test_155.png',
               './images/post processing difference/images/test_169.png', 
               './images/post processing difference/images/test_169.png']
mask_paths = ['./images/post processing difference/results/test_155_no_post_processing.png', 
              './images/post processing difference/results/test_155_no_post_processing.png',
              './images/post processing difference/results/test_169_no_post_processing.png', 
              './images/post processing difference/results/test_169_no_post_processing.png', 
              './images/post processing difference/results/test_155_post_processing.png', 
              './images/post processing difference/results/test_155_post_processing.png',
              './images/post processing difference/results/test_169_post_processing.png', 
              './images/post processing difference/results/test_169_post_processing.png']

xlabels = ['', 
           '', 
           '', 
           '', 
           '', 
           '', 
           '', 
           '']
ylabels = ['no post-processing', 
           '', 
           '', 
           '', 
           'post-processing', 
           '', 
           '',
           '']


def normalize(img):
    if np.max(img) > 1.0:
        return img / 255.0
    else:
        return img


def patch_to_label(patch, thresh):
    df = np.mean(patch)
    if df > thresh:
        return 1
    else:
        return 0


def mask_to_submission(img, mask, thresh=0.5, patch_size=16, color=[0.3, 0.0, 0.0]):
    overlay = np.zeros(img.shape)
    for j in range(0, img.shape[1], patch_size):
        for i in range(0, img.shape[0], patch_size):
            patch = mask[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch, thresh)
            overlay[i:i + patch_size, j:j + patch_size, 0] = color[0] * label
            overlay[i:i + patch_size, j:j + patch_size, 1] = color[1] * label
            overlay[i:i + patch_size, j:j + patch_size, 2] = color[2] * label
    return np.minimum(img + overlay, 1.0)


color = [0.3, 0.0, 0.0]
columns = 4
rows = 2
fig = plt.figure(figsize=(15.0 / rows, 15.0 / columns))
plt.title(' ')
plt.axis('off')
plt.tight_layout()

axs = []

for i in range(1, columns * rows, 2):

    img = normalize(io.imread(image_paths[i], as_gray=False))
    img = normalize(img)
    mask = io.imread(mask_paths[i], as_gray=True)
    #mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask = normalize(mask)

    overlay = mask_to_submission(img, mask, color=[0.3, 0.0, 0.0])

    ax = fig.add_subplot(rows, columns, i+1)
    axs.append(ax)
    plt.imshow(overlay)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.15, hspace=None)
    ax.set_ylabel(ylabels[i], fontsize=10, fontweight='normal')
    ax.set_xlabel(xlabels[i], fontsize=10, fontweight='normal')
    ax.xaxis.set_label_position('top')


for i in range(0, columns * rows, 2):
    mask = io.imread(mask_paths[i], as_gray=True)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask = normalize(mask)
    ax = fig.add_subplot(rows, columns, i+1)
    axs.append(ax)
    plt.imshow(mask)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.15, hspace=None)
    ax.set_ylabel(ylabels[i], fontsize=10, fontweight='normal')
    ax.set_xlabel(xlabels[i], fontsize=10, fontweight='normal')
    ax.xaxis.set_label_position('top')

fig.savefig(output_path, dpi=150)
fig.savefig(output_path.split('.')[-2] + '.png', dpi=150)
