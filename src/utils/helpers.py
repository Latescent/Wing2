"""A collection of tools that are required to help other scripts.
"""

import csv
import cv2
import numpy as np
import os
import sys

def progress_bar(total: int, current: int, prefix: str = "", suffix: str = "") -> None:
    """Displays a text-based rogress bar.

    Creates a text based progress bar. The effects are achieved with terminal text manipulation.

    Args: 
        total : The total number of tasks, representing 100% of the bar.
        current : The number of tasks that have been completed.
        prefix : An optional string to display at the beginning of the progress bar.
        suffix : An optional string to display at the end of the progress bar.

    Returns:
        None
    """
    bar_length = 50
    progress = float(current) / total
    arrow = '=' * int(round(progress * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write(f"\r{prefix} [{arrow}{spaces}] {int(progress*100)}% {suffix}")
    sys.stdout.flush()
    
    if current == total:
        sys.stdout.write("\n")


def read_csv(file_path):
    """
    Reads a CSV file and converts it into a list of lists.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of lists, where each inner list contains the
              parameters from a row in the CSV file. Returns an empty
              list if the file is not found or an error occurs.
    """
    data_list = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as csv_file:
            # Create a csv reader object
            csv_reader = csv.reader(csv_file)
            
            # Convert the reader object to a list of lists
            data_list = list(csv_reader)
            
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return data_list


def noise_level_detection(image_dir):
    image = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError
    
    # Use adaptive thresholding for local binarization
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
    mean_value = np.mean(thresh)

    return (os.path.basename(image_dir), float(mean_value))

