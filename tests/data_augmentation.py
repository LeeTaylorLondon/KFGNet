"""
Author: Lee Taylor

data_augmentation.py : test functionality for data augmentation
"""
from functions import dataloader
import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_frames(data_):
    """ Plot 32 frames to check correct slice is unpacked. """
    fig = plt.figure(figsize=(10, 10))

    for i in range(32):
        fig.add_subplot(8, 4, i + 1)  # Assuming grid of 8x4 for 32 images
        plt.imshow(data_[0, 0, i], cmap='gray')
        plt.axis('off')

    plt.show()


def frames_to_mp4(data_, output_path='video.mp4'):
    """ Convert 32 frames to an .mp4 video file to play back. """
    # Get video dimensions
    height, width = data_.shape[-2:]

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    # Write frames to video
    for i in range(data_.shape[2]):
        frame = cv2.normalize(data_[0, 0, i], None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video.write(frame)

    # Release the VideoWriter
    video.release()


def data_augment(data_):
    """ Modify 32 frames of ultrasound data_ similarly. """
    # As an example, let's do a simple normalization
    data_normalized = data_ / np.max(data_)

    return data_normalized


if __name__ == '__main__':
    # Load 32 frames as one numpy array from dataloader
    data = None
    for i, (frames, labels) in enumerate(dataloader()):
        data = frames
        if i == 0:
            break

    data = data.numpy()
    print(f"videos, channels, frames, width, height")
    print(data.shape)
    print(type(data))
    """
    >>> videos, channels, frames, width, height
    >>> (1, 3, 32, 112, 112)
    >>> <class 'numpy.ndarray'>
    """

    frames_to_mp4(data, "test_run1.mp4")

    # Mark EOF
    pass
