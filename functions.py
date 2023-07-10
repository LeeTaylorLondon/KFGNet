"""
Author: Lee Taylor

functions.py : contains functions to support training and testing the model
"""
import cv2
import torch
import numpy as np
from os import listdir


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

    sequence = video_data[start_index:end_index]

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


def dataloader():
    """
    Organise data for the training pipeline.
    :yield: input (1, 32, 112, 112, 3), labels [1.0, 0.0]
    """
    # Load image and video dictionaries
    image_dict = return_image_fns_dict()
    video_dict = return_video_fns_dict()
    prefix = "data/data"

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
        if video_data is None:
            video_data = unpack_video(f'../data/data/videos/{video_name}')
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


if __name__ == '__main__':
    pass
