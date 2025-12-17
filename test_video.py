import os
import numpy as np 

import matplotlib.pyplot as plt
import matplotlib.animation as animation 

from src import video_utils, motion, img_utils
videos     = video_utils.VideoSource('data/raw_video/Road traffic video for object recognition.mp4', n_slices=2)
images     = next(videos)
images_bg  = list(map( img_utils.rgb_to_greyscale, images ))
velocities = motion.velocities_from_video(images_bg)

fig = plt.figure(figsize = (4, 6))

third = 1/3

ax1 = plt.axes([0, 0, 1, third])
ax2 = plt.axes([0, third, 1, third])
ax3 = plt.axes([0, 2*third, 1, third])
for ax in [ax1, ax2, ax3]:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ['left', 'right', 'top', 'bottom']:
        ax.spines[side].set_visible(False)


top_image    = ax3.imshow( images[0] )
middle_image = ax2.imshow( velocities[0], cmap='coolwarm' )
bottom_image = ax1.imshow( velocities[1], cmap='coolwarm' )


def updatefig(*args):
    images     = next(videos)
    images_bg  = list(map( img_utils.rgb_to_greyscale, images ))
    velocities = motion.velocities_from_video(images_bg)
    top_image.set_array( images[0] )
    middle_image.set_array( velocities[0] )
    bottom_image.set_array( velocities[0] )
    return top_image, middle_image, bottom_image

ani = animation.FuncAnimation(fig, updatefig, interval=0, blit=True)

plt.show()
