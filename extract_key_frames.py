"""
Author: Lee Taylor
"""
import cv2
import numpy as np
import pandas as pd
import os


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

        # Append the frame (a numpy array) to the list of frames.
        frames.append(frame)

    # Close the video file.
    cap.release()

    # Convert the list of frames to a single numpy array.
    video_data = np.stack(frames)

    return video_data


def get_single_frame(video_data, frame_index):
    """
    Given a video stored in a numpy array, select a single frame
    at the specified index.

    Args:
        video_data (numpy.array): The video data.
        frame_index (int): The index of the frame to select.

    Returns:
        frame (numpy.array): The selected frame.
    """
    # Make sure the frame_index is within the video length
    if frame_index < 0 or frame_index >= len(video_data):
        print(f"Error: frame_index {frame_index} out of range for video of length {len(video_data)}")
        return None

    # Get the frame at the desired index
    frame = video_data[int(frame_index)]

    return frame


def video_keyframe():
    """
    Yield video number, key frame number.
    """
    # Read data
    file = "data/data/avi.xlsx"
    df = pd.read_excel(file)
    columns = df[['d', 'g', 'j', 'm']]

    # Iterate over columns with video numbers and key frame indexes
    for index, row in columns.iterrows():
        for item in row:
            if not str(item) == 'nan':
                # Out info
                # print(f"#Video: {index}, Key Frame: {item}")
                yield index, item


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


def process_videos():
    """
    For each yield from the video_keyframe function, opens the video
    and gets the key frame. The video number corresponds to the video
    name from the video_names function.
    """
    # Get the dictionary of video names
    video_names_dict = get_video_names_dict()

    if not os.path.exists("data/data/key_frames"):
        os.makedirs("data/data/key_frames")

    # For each yield from the video_keyframe function...
    for video_num, key_frame in video_keyframe():
        # Get the name of the video.
        video_name = video_names_dict[str(video_num)]
        video_path = os.path.join("data/data/videos", video_name)

        # Unpack the video into frames.
        video_data = unpack_video(video_path)
        if video_data is None:
            print(f"video_data skip = {video_data}")
            continue  # Skip to next video if current video couldn't be opened.

        # Get the key frame sequence.
        frame = get_single_frame(video_data, key_frame)

        # Out information
        print(f"Video_num: {video_num}, Video: {video_name}")
        # print(f"Video: {video_name}, Key Frame: {key_frame}, Sequence shape: {frame.shape}")

        # Save the frame to a .jpg image.
        output_file = f"data/data/key_frames/{video_num}_{int(key_frame)}_{video_name.split('_')[-1][:1]}.jpg"
        cv2.imwrite(output_file, frame)


if __name__ == '__main__':
    # print(video_names())
    process_videos()
