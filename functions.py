"""
Author: Lee Taylor

functions.py : contains functions to support training and testing the model
"""
import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from os import listdir
from collections import deque
import scipy.ndimage as ndimage
from itertools import zip_longest


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def unpack_video(video_path):
    """
    Unpack a video into numpy arrays representing the pixels in each frame.
    """
    # Open the video file.
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened correctly.
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None

    frames = []

    while True:
        # Read the next frame.
        ret, frame = cap.read()

        # If the frame was not read correctly, we have reached the end of the video.
        if not ret:
            break

        # Resize the frame.
        frame = cv2.resize(frame, (112, 112))

        # Append the frame (a numpy array) to the list of frames.
        frames.append(frame)

    # Close the video file.
    cap.release()

    # Convert the list of frames to a single numpy array.
    video_data = np.stack(frames)

    return video_data


def get_continuous_frames(video_data, frame_index=94, T=32):
    """
    Given a video stored in a numpy array, selects T continuous frames
    from the frame index where the index frame becomes the center of the
    selected sequence. If T is even, the center will be the frame at index
    `frame_index - T//2 + 1`.

    If the frame_index is too close to the start or end of the video for
    the sequence to fit, the sequence will start or end with the video
    respectively.

    Args:
        video_data (numpy.array): The video data.
        frame_index (int): The frame index.
        T (int): The number of frames to select.

    Returns:
        sequence (numpy.array): The selected sequence of frames.
    """
    half_T = T // 2
    start_index = max(0, frame_index - half_T + T % 2)
    end_index = min(len(video_data), start_index + T)
    start_index = max(0, end_index - T)  # adjust start if end is beyond video length

    # print(f"DEBUG: start_index = {start_index}, end_index = {end_index}")
    sequence = video_data[int(start_index):int(end_index)]

    return sequence


def return_image_fns_dict(sanity_check=False):
    """
    Sort frame name into dictionary/object
    Video name, continuous frames, etc.etc
    :return: dict ->
    ('0', {'image_no': '0', 'video_no': '1', 'scan_dir': 'r', 'view_': 't',
            'y_actual': 'b', 'temporal_index': '94', 'video_filename': '1_r_t_b.avi'})
    ('10', {'image_no': '10', 'video_no': '1', 'scan_dir': 'r', 'view_': 't',
            'y_actual': 'b', 'temporal_index': '89', 'video_filename': '1_r_t_b.avi'})
    ...
    ('9', {'image_no': '9', 'video_no': '0', 'scan_dir': 'l', 'view_': 't',
            'y_actual': 'b', 'temporal_index': '40', 'video_filename': '0_l_t_b.avi'})
    """
    try:
        image_fns = listdir("data/data/images")
    except FileNotFoundError as e:
        image_fns = listdir("../data/data/images")
    image_fns_dict = {}
    # Loop over image file names
    for image_fn in image_fns:
        # Only process .jpg files
        if image_fn.__contains__(".json"):
            continue
        # Preprocess frame information from frame name
        numbers = image_fn.replace('.jpg', '')
        n = numbers = numbers.split('_')
        # Acquire frame information from frame name
        temp_dict = {"image_no": numbers[0],
                     "video_no": numbers[1],
                     "scan_dir": numbers[2],
                     "view_": numbers[3],
                     "y_actual": numbers[4],
                     "temporal_index": numbers[5],
                     "video_filename": f"{n[1]}_{n[2]}_{n[3]}_{n[4]}.avi"}
        image_fns_dict.update({numbers[0]: temp_dict.copy()})
    # Dictionary sanity check
    if sanity_check:
        print("---[image_fns_dict]---")
        for item in image_fns_dict.items():
            print(item)
        print()
    # End function
    return image_fns_dict


def return_video_fns_dict(sanity_check=False):
    """
    Convert video file .avi into frames
    Extract 30 to 32 frames near keyframe using temporal index

    :return: dict ->
    ('0', {'video_no': '0', 'scan_dir': 'r', 'view_': 't', 'y_actual': 'b'})
    ...
    ('9', {'video_no': '9', 'scan_dir': 'l', 'view_': 't', 'y_actual': 'b'})
    """
    try:
        video_fns = listdir("data/data/videos")
    except FileNotFoundError as e:
        video_fns = listdir("../data/data/videos")
    video_fns_dict = {}
    # Loop over image file names
    for video_fn in video_fns:
        # Only process .jpg files
        if video_fn.__contains__(".json"):
            continue
        # Preprocess frame information from frame name
        numbers = video_fn.replace('.avi', '')
        numbers = numbers.split('_')
        # Acquire frame information from frame name
        temp_dict = {
            "video_no": numbers[0],
            "scan_dir": numbers[1],
            "view_": numbers[2],
            "y_actual": numbers[3]
        }
        video_fns_dict.update({numbers[0]: temp_dict.copy()})
    # Dictionary sanity check
    if sanity_check:
        print("---[video_fns_dict]---")
        for item in video_fns_dict.items():
            print(item)
        print()
    return video_fns_dict


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
    format (frames, width, height, channels)
    """
    if random.choice([True, False]):
        volume = volume[::-1, :, :, :]  # flip along frames
    if random.choice([True, False]):
        volume = volume[:, ::-1, :, :]  # flip along width
    if random.choice([True, False]):
        volume = volume[:, :, ::-1, :]  # flip along height

    return volume


def random_rotation_3d(volume, max_angle):
    """
    Randomly rotate volume along frames dimension. Volume input should be in
    format (frames, width, height, channels)
    """
    angle = random.uniform(-max_angle, max_angle)
    volume_rot = np.empty(volume.shape, dtype=np.float32)

    # Apply rotation to each frame
    for i in range(volume.shape[0]):
        for j in range(volume.shape[3]):
            volume_rot[i, :, :, j] = ndimage.rotate(volume[i, :, :, j], angle, axes=(0, 1), reshape=False)

    return volume_rot


def random_intensity_shift(volume, max_offset, max_scale_delta):
    """
    Randomly shift intensity of volume. Volume input should be in
    format (frames, width, height, channels)
    """
    offset = random.uniform(-max_offset, max_offset)
    scale = random.uniform(1 - max_scale_delta, 1 + max_scale_delta)

    volume = volume.astype('float64')  # Convert volume to float64
    volume += offset
    volume *= scale

    return volume


class MyDataset(torch.utils.data.IterableDataset):
    """
    Yield data for training purposes.
    """
    def __init__(self, generator_function):
        super(MyDataset).__init__()
        self.generator_function = generator_function

    def __iter__(self):
        return self.generator_function()


def dataloader_test():
    """
    Organise data for the training pipeline.
    :yield: input (1, 32, 112, 112, 3), labels [1.0, 0.0]
    """
    # Load image and video dictionaries
    image_dict = return_image_fns_dict()
    video_dict = return_video_fns_dict()

    test_set_numbers = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    all_numbers = [x for x in range(50)]
    for number in all_numbers:
        if number not in test_set_numbers:
            try:
                del image_dict[str(number)]
            except KeyError as e:
                pass

    print(f"image_dict.keys() = {list(image_dict.keys())}")

    # for item in list(image_dict.values())[:1]:
    #     print(video_dict[item["video_no"]])
    #     print('_'.join(list(video_dict[item["video_no"]].values())) + '.avi')
    """ 
    Write a function using an image filename to get the corresponding video
    and unpack the 32 frames using the key frame from the image filename dictionary.

    image_number, image_key_frame_index -> video_number -> 32_frames
    """
    for item in image_dict.values():
        # item = {'image_no': '0', 'video_no': '1', ..., 'temporal_index': '94', 'video_filename': '1_r_t_b.avi'}
        video_name = '_'.join(list(video_dict[item["video_no"]].values())) + '.avi'
        video_data = unpack_video(f'data/data/videos/{video_name}')
        data = get_continuous_frames(video_data, frame_index=int(item["temporal_index"]))
        data = np.expand_dims(data, axis=0)
        data = np.transpose(data, (0, 4, 1, 2, 3))
        # Calculate labels, b = benign, m = malignant, [b, m] -> if b : [1, 0] ? [0, 1]
        labels = [0.0, 0.0]
        if item["y_actual"] == 'b':
            labels[0] = 1.0
        else:
            labels[1] = 1.0
        labels = torch.tensor(labels)
        # Pass C3D input and labels
        data = torch.from_numpy(data).float()
        yield data, labels


def video_names():
    """
    :return: list of video names
    """
    path = "data/data/videos"
    return os.listdir(path)


def get_video_names_dict():
    """
    Return a dictionary which contains key video number and video name.
    """
    rv = {}
    names = video_names()
    for i in range(50):
        for name in names:
            # if name.__contains__(str(i)):
            if name.split('_')[0] == str(i):
                rv.update({str(i): name})
                continue
    return rv


def video_keyframe():
    """
    Yield video number, key frame number.
    """
    # Read data
    file = "data/data/avi.xlsx"
    df = pd.read_excel(file)
    columns = df[['d', 'g', 'j', 'm']]

    # Get the dictionary of video names {'0': '0_r_t_b.avi', ...}
    video_names_dict = get_video_names_dict()

    # Skip videos belonging to the test set
    test_set_numbers = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    for number in test_set_numbers:
        try:
            del video_names_dict[str(number)]
        except KeyError:
            pass

    # Sort the video_keys into class 1 (malignant) and class 0 (benign)
    class_1_keys = []
    class_0_keys = []
    for index, row in columns.iterrows():
        if index in test_set_numbers:
            continue
        for item in row:
            if not str(item) == 'nan':
                video_name = video_names_dict.get(str(index), '')
                if video_name.split('_')[-1][0] == 'b':
                    class_0_keys.append((index, item))
                elif video_name.split('_')[-1][0] == 'm':
                    class_1_keys.append((index, item))

    # Calculate the difference in lengths
    length_diff = abs(len(class_0_keys) - len(class_1_keys))

    for _ in range(length_diff):
        random_var = random.randint(0, len(class_1_keys) - 1)
        class_1_keys.append(class_1_keys[random_var])

    for _ in range(3):
        class_1_keys.extend(class_1_keys)
        class_0_keys.extend(class_0_keys)

    print(f"len(c1) = {len(class_1_keys)}")
    print(f"len(c0) = {len(class_0_keys)}")

    # Iterate through class_1_keys and class_0_keys in an alternating fashion
    class_1_keys_iter = iter(class_1_keys)
    class_0_keys_iter = iter(class_0_keys)
    while True:
        try:
            yield next(class_1_keys_iter)
        except StopIteration:
            break
        try:
            yield next(class_0_keys_iter)
        except StopIteration:
            break


def dataloader_augment():
    """
    For each yield from the video_keyframe function, opens the video
    and gets the 32-frame sequence. The video number corresponds to the video
    name from the video_names function.
    :yield: input (1, 32, 112, 112, 3), labels [1.0, 0.0]
    """
    # Get the dictionary of video names {'0': '0_r_t_b.avi', ...}
    video_names_dict = get_video_names_dict()

    # Skip videos belonging to the test set
    test_set_numbers = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    for number in test_set_numbers:
        try:
            del video_names_dict[str(number)]
        except KeyError:
            pass

    for video_num, key_frame in video_keyframe():
        # Get the name of the video.
        try:
            video_name = video_names_dict[str(video_num)]
        except KeyError:
            continue
        video_path = os.path.join("data/data/videos", video_name)

        # Unpack the video into frames.
        video_data = unpack_video(video_path)
        if video_data is None:
            print(f"video_data skip = {video_data}")
            continue  # Skip to next video if current video couldn't be opened.

        # Get the 32-frame sequence.
        data = get_continuous_frames(video_data, frame_index=key_frame)

        # Calculate labels, b = benign, m = malignant, [b, m] -> if b : [1, 0] ? [0, 1]
        labels = [0.0, 0.0]
        # print(f"{video_name.split('_')[-1][0]} = video_name.split('_')[-1][0]")
        if video_name.split('_')[-1][0] == 'b':
            labels[0] = 1.0
        else:
            labels[1] = 1.0
        labels = torch.tensor(labels)

        # Apply random modifications and iterate over the same 32 frames 5 times
        modified_data = data_augment(data)
        modified_data = np.transpose(modified_data, (3, 0, 1, 2))
        # Normalize data to be between 0 and 1
        modified_data = (modified_data - np.min(modified_data)) / (np.max(modified_data) - np.min(modified_data))
        modified_data = torch.from_numpy(modified_data.copy()).float()
        yield modified_data, labels


if __name__ == '__main__':
    pass
