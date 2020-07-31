import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os

output_path = 'comparison_dilation_plot.pdf'

image_paths = ['./images/dilation difference/images/test_107.png', 
               './images/dilation difference/images/test_151.png', 
               './images/dilation difference/images/test_201.png',
               './images/dilation difference/images/test_107.png', 
               './images/dilation difference/images/test_151.png', 
               './images/dilation difference/images/test_201.png']
mask_paths = ['./images/dilation difference/results/test_107_unet.png', 
              './images/dilation difference/results/test_151_unet.png', 
              './images/dilation difference/results/test_201_unet.png', 
              './images/dilation difference/results/test_107_unet_dilated_v2_transposed.png', 
              './images/dilation difference/results/test_151_unet_dilated_v2_transposed.png', 
              './images/dilation difference/results/test_201_unet_dilated_v2_transposed.png']

xlabels = ['', 
           '', 
           '', 
           '', 
           '', 
           '']
ylabels = ['no dilation', 
           '', 
           '', 
           'dilation', 
           '', 
           '']


def normalize(img):
    if np.max(img) > 1.0:
        return img / 255.0
    else:
        return img


color = [0.3, 0.0, 0.0]
columns = 3
rows = 2
fig = plt.figure(figsize = (15.0 / rows,15.0 / columns))
plt.title(' ')
plt.axis('off')
plt.tight_layout()

axs = []

for i in range(0, columns * rows):
    img = normalize(io.imread(image_paths[i], as_gray=False))
    img = normalize(img)
    mask = io.imread(mask_paths[i], as_gray=True)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask = normalize(mask)

    overlay = np.minimum(img + mask * color, 1.0)

    ax = fig.add_subplot(rows, columns, i+1)
    axs.append(ax)
    plt.imshow(overlay)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    plt.subplots_adjust(wspace=-0.15, hspace=None)

    ax.set_ylabel(ylabels[i], fontsize=12, fontweight='normal')
    ax.set_xlabel(xlabels[i], fontsize=12, fontweight='normal')
    ax.xaxis.set_label_position('top')

fig.savefig(output_path, dpi=150)
fig.savefig(output_path.split('.')[-2] + '.png', dpi=150)
