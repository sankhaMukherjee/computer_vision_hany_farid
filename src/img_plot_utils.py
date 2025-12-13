import matplotlib.pyplot as plt

def plot_image(image, filename=None):

    y, x, _ = image.shape
    fig = plt.figure(figsize=(3, y*3/x))
    ax = plt.axes([0,0,1,1])
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(False)

    if filename is not None:
        plt.savefig(filename, dpi=300)

    return fig