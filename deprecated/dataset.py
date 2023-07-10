import torch
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
import random
from scipy.ndimage.interpolation import rotate
# Get file names
from os import listdir
# ImportError: cannot import name 'generate_sim' ...
# from utils import generate_sim


class VideoDataset(Dataset):
    """
    Process video files.
    """
    def __init__(self, file_names, transform=True, resize=(112, 112)):
        super(VideoDataset, self).__init__()
        self.data = file_names
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        volume = []

        volume_path = self.data[index]

        for img_path in glob.glob(volume_path + '/*.jpg'):
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_CUBIC)
            volume.append(img)

        volume = np.array(volume).astype(np.float32)

        # ImportError / NameError: ...
        # sim = generate_sim(volume)

        volume /= 255

        if self.transform:
            if random.choice([True, False]):
                volume = random_intensity_shift(volume, 0.1, 0.1)

            if random.choice([True, False]):
                volume = random_flip_3d(volume)

        if volume_path.split('_')[-1] == 'm':
            label = 1
        else:
            label = 0

        print(f"volume.shape = {volume.shape}")
        volume = torch.Tensor(volume).permute(3, 0, 1, 2)
        print(f"volume.shape = {volume.shape}")
        # print(volume)
        # print(label)
        # return volume, label, sim
        return volume, label


def random_flip_3d(volume):
    """
    Data augmentation.
    :param volume:
    :return:
    """
    if random.choice([True, False]):
        volume = volume[::-1, :, :].copy()  # here must use copy(), otherwise error occurs
    if random.choice([True, False]):
        volume = volume[:, ::-1, :].copy()
    if random.choice([True, False]):
        volume = volume[:, :, ::-1].copy()

    return volume


def random_rotation_3d(volume, max_angles):
    """
    Data augmentation.
    :param volume:
    :param max_angles:
    :return:
    """
    volume1 = volume
    # rotate along x-axis
    angle = random.uniform(-max_angles[2], max_angles[2])
    volume2 = rotate(volume1, angle, order=2, mode='nearest', axes=(0, 1), reshape=False)

    # rotate along y-axis
    angle = random.uniform(-max_angles[1], max_angles[1])
    volume_rot = rotate(volume2, angle, order=2, mode='nearest', axes=(0, 2), reshape=False)

    return volume_rot


def random_intensity_shift(volume, max_offset, max_scale_delta):
    """
    Data augmentation.
    :param volume:
    :param max_offset:
    :param max_scale_delta:
    :return:
    """
    offset = random.uniform(-max_offset, max_offset)
    scale = random.uniform(1 - max_scale_delta, 1 + max_scale_delta)

    volume = volume.copy()
    volume += offset
    volume *= scale

    return volume


if __name__ == '__main__':
    # Does not work - because of permute is wrong
    filenames = listdir("data/data/videos")

    # Ensure dataset.py is useable # file_names, transform=True, resize=(112, 112)
    video_dataset_object = vdo = VideoDataset(file_names=filenames, transform=False)

    print(video_dataset_object)
    print(f"vdo.len = {vdo.__len__()}\n"
          f"vdo.getitem(0) = {vdo.__getitem__(0)}")

    # Mark EOF
    pass

