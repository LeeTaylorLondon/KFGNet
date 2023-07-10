"""
Author: Lee Taylor

data_augmentation.py : test functionality for data augmentation
"""
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from functions import dataloader


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
    """ Randomly apply one of the augmentation techniques to the 32 frames of ultrasound data_."""
    # Randomly select an augmentation method
    augment_methods = [random_flip_3d, random_rotation_3d, random_intensity_shift]
    augment_method = random.choice(augment_methods)

    data_augmented = None
    # Apply selected augmentation method
    if augment_method == random_flip_3d:
        data_augmented = augment_method(data_)
    elif augment_method == random_rotation_3d:
        max_angle = 20  # You can adjust this parameter
        data_augmented = augment_method(data_, max_angle)
    elif augment_method == random_intensity_shift:
        max_offset = 0.1  # You can adjust this parameter
        max_scale_delta = 0.2  # You can adjust this parameter
        data_augmented = augment_method(data_, max_offset, max_scale_delta)

    return data_augmented


def random_flip_3d(volume):
    """
    Randomly flip volume across different dimensions. Volume input should be in
    format (videos, channels, frames, width, height)
    """
    if random.choice([True, False]):
        volume = volume[:, :, ::-1, :, :]  # flip along frames
    if random.choice([True, False]):
        volume = volume[:, :, :, ::-1, :]  # flip along width
    if random.choice([True, False]):
        volume = volume[:, :, :, :, ::-1]  # flip along height

    return volume


def random_rotation_3d(volume, max_angle):
    """
    Randomly rotate volume along frames dimension. Volume input should be in
    format (videos, channels, frames, width, height)
    """
    angle = random.uniform(-max_angle, max_angle)
    volume_rot = np.empty(volume.shape, dtype=np.float32)

    # Apply rotation to each video
    for i in range(volume.shape[0]):
        for j in range(volume.shape[1]):
            volume_rot[i, j] = ndimage.rotate(volume[i, j], angle, axes=(1, 3), reshape=False)

    return volume_rot


def random_intensity_shift(volume, max_offset, max_scale_delta):
    """
    Randomly shift intensity of volume. Volume input should be in
    format (videos, channels, frames, width, height)
    """
    offset = random.uniform(-max_offset, max_offset)
    scale = random.uniform(1 - max_scale_delta, 1 + max_scale_delta)

    volume += offset
    volume *= scale

    return volume


if __name__ == '__main__':
    # Load 32 frames as one numpy array from dataloader
    data = None
    for i_, (frames, labels) in enumerate(dataloader()):
        data = frames
        if i_ == 0:
            break

    # Convert 32 frames from Torch to Numpy
    data = data.numpy()

    # Debug info
    print(f"videos, channels, frames, width, height")
    print(data.shape)
    print(type(data))
    """
    >>> videos, channels, frames, width, height
    >>> (1, 3, 32, 112, 112)
    >>> <class 'numpy.ndarray'>
    """

    # Test creating a video from 32 frames
    # frames_to_mp4(data, "test_run1.mp4")

    # Mark EOF
    pass
