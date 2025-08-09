"""A collection of tools that are required to help other scripts."""

import csv
import cv2
import numpy as np
import os
import sys


def calculate_area(image):
    """
    This function reads a wing image, finds the largest area (the wing),
    calculates its area, and creates an output image with the area colored.

    Args:
        image (np.ndarray): Path to the input image file.

    Returns:
        float: The calculated area in square pixels.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wing_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(wing_contour)

    return area


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
    arrow = "=" * int(round(progress * bar_length) - 1) + ">"
    spaces = " " * (bar_length - len(arrow))

    sys.stdout.write(f"\r{prefix} [{arrow}{spaces}] {int(progress * 100)}% {suffix}")
    sys.stdout.flush()

    if current == total:
        sys.stdout.write("\n")


def noise_level_detection(image_dir: str):
    """
    Takes the mean value of the pixel colors of a black and white image.

    Args:
        image_dir (str): Path to the image.

    Returns:
        tuple[str, float]: A tuple that contains the image name and the brightness value.

    Raises:
        ValueError if image doesn't exist.
    """
    image = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError

    # Use adaptive thresholding for local binarization
    thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )

    mean_value = np.mean(thresh)  # type: ignore
    return (os.path.basename(image_dir), float(mean_value))


def string_to_tuple(input: str) -> tuple:
    """
    Takes a tuple in the form of a string and returns the original tuple

    Args:
        input (str): A string containing a tuple

    Returns:
        tuple[int]: The tuple inside the input string
    """
    params = input.replace("(", "").replace(")", "").split(",")
    result = [int(item) for item in params]
    return tuple(result)


def tuple_to_string(input: tuple) -> str:
    return f"({input[0]},{input[1]})"


def load_csv(dir):
    """
    Reads a CSV file and converts it into a list of lists.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        list: A list of lists, where each inner list contains the
              parameters from a row in the CSV file. Returns an empty
              list if the file is not found or an error occurs.
    """

    def format_str(S: str):
        """Converts strings in the csv to the correct format"""
        # Check to see if it is a tuple
        S.strip()
        if "(" and ")" in S:
            # Remove the unnecessary punctuation and split the tuple
            temp: list = S.replace("(", "").replace(")", "").replace(" ", "").split(",")

            # Try to convert the strings to floating values if possible
            for index in range(len(temp)):
                try:
                    if temp[index].isdigit():
                        temp[index] = int(temp[index])
                    else:
                        temp[index] = float(temp[index])
                except Exception:
                    pass

            # Convert temp into a tuple and return it
            return tuple(temp)

        elif S.isdigit():
            return int(S)

        elif S.replace(".", "").isdigit():
            return float(S)

        else:
            return S

    def row_reader(row: list):
        """Parses each line from the csv file into a dictionary with the following format
        { Folder_name : { Noise_level : [arg_list] } }
        """
        # Create a base dict
        data = dict()

        # Create and store: folder name, noise level
        folder_name = format_str(row[0])
        noise_lvl = format_str(row[1])

        # Add the structure
        data[folder_name] = dict()

        # Create the value list
        value_list = list()
        for index in range(2, len(row)):
            value_list.append(format_str(row[index]))

        # Add the value list to the data
        data[folder_name][noise_lvl] = value_list

        return data

    try:
        with open(dir, "r") as csv_file:
            # Read the csv file
            csv_reader = csv.reader(csv_file)

            # Create the data dict\
            data = dict()

            # Iterate the csv file
            for row in csv_reader:
                data |= row_reader(row)

            print("Loading CSV: Completed")

            # Return the data
            return data

    except Exception as E:
        print(f"Error in loading the csv: {E}")
        exit(1)
