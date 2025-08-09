import csv
import concurrent.futures
import os

from pathlib import Path
from preprocessing.filter import process_bee_wing
from utils.helpers import load_csv
from utils.helpers import noise_level_detection
from utils.helpers import progress_bar

error_counter = 0
image_counter = 0


def load_images(
    image_dir: str, output_dir: str, verbose: bool = True
) -> tuple[list[str], list[str]]:
    """Looks inside the received directories for images

    Args:
        image_dir(str): Directory if the images that are not yet processed
        output_dir(str): Directory of the images that are already processed
        verbose(bool, optional): Displays the event log

    Returns:
        tuple[list[str], list[str]]: Two lists of the names of the images in both directories
    """
    try:
        loaded_image_list = [
            os.path.join(image_dir, img)
            for img in os.listdir(image_dir)
            if img.endswith((".png", ".jpg", ".jpeg"))
        ]
        loaded_image_list.sort()

        processed_image_list = [
            os.path.join(output_dir, img)
            for img in os.listdir(output_dir)
            if img.endswith((".png", ".jpg", ".jpeg"))
        ]
        processed_image_list.sort()

        if verbose:
            print("Loading images: Completed")

    except Exception as E:
        print(f"Error in load_images: {E}")
        exit(1)

    return loaded_image_list, processed_image_list


def filter_all_images(input_dir, csv_dir, output_dir):
    """Skeletonizes multiple images at the same time
    The images should be named "[selection]-[index|metadata]"

    Args:
        input_dir(str): Directory of the non-processed images
        csv_dir(str): Directory of the parameters file in csv format
        output_dir(str): Directory of the processed images

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    images_list, processed_list = load_images(input_dir, output_dir)
    desired_images = [image for image in images_list if image not in processed_list]
    data = load_csv(csv_dir)

    # Process one image(Parallel processing compatible)
    def process_one_image(image_dir):
        try:
            noise_lvl = noise_level_detection(image_dir)[1]
            folder_name = os.path.basename(image_dir)[:2]
            # image_data = {folder_name : {noise_lvl : []}}
            arg = (
                data[folder_name].get(noise_lvl)
                or data[folder_name][
                    min(data[folder_name], key=lambda key: abs(key - noise_lvl))
                ]
            )
            global image_counter
            progress_bar(
                len(desired_images), image_counter, prefix="Skeletonization progress: "
            )
            image_counter += 1
            process_bee_wing(image_dir, arg, output_dir)
        except Exception as E:
            print(f"Error in processing {image_dir}")

            global error_counter

            error_counter += 1

            # Log every failed action in log.txt
            with open("log.txt", "a") as F:
                F.write(f"Error loading image: {os.path.basename(image_dir)}\n\t{E}\n")
                try:
                    F.write(f"\tNoise level: {noise_lvl}\n")  # type: ignore
                except Exception as E1:
                    F.write(f'\tLogging "noise_lvl" failed: {E1}\n')

                try:
                    F.write(f"\tFolder name: {folder_name}\n")  # type: ignore
                except Exception as E2:
                    F.write(f'\tLogging "folder_name" failed: {E2}\n')

                try:
                    F.write(f"\tArguments: {arg}\n")  # type: ignore
                except Exception as E3:
                    F.write(f'\tLogging "Arguments" failed: {E3}\n')

                F.write(
                    "\n----------------------------------------------------------------------------------------------------\n"
                )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(process_one_image, desired_images))

    global error_counter

    print(f"All images processed with {error_counter} errors")


if __name__ == "__main__":
    """Skeletonizes multiple images at the same time
    The images should be named "[selection]-[index|metadata]"

    Args:
        verbose(bool, optional): If true, the program log will be displayed

    Returns:
        None
    """
    home_directory = Path(__file__).parent.parent
    source_dir = f"{home_directory}/data/raw_images"
    csv_dir = f"{home_directory}/configs/filter_params.csv"
    skeletonized_dir = f"{home_directory}/data/skeletonized_images"

    filter_all_images(source_dir, csv_dir, skeletonized_dir)
