import os, cv2, sys, numpy as np, csv
import concurrent.futures
from filter import process_bee_wing, counter_lock, counter

error_counter = 0
image_counter = 0

def progress_bar(len, counter, txt="Loading:"):
    percentage = int(counter * 100 / len)
    print(f"{txt} |{'='*percentage}{'-'*(100-percentage)}| {percentage}% | {counter}", end='\r')
    if counter == len:
        sys.stdout.write(f"\r{txt} 100%\033[K\n")

# Function that marges two dictionaries
def deep_merge(dict1, dict2):
    # Merges dict2 into dict1 recursively
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                dict1[key] = deep_merge(dict1[key], value)
            else:
                # Overwrite with dict2's value
                dict1[key] = value
        else:
            # Add new key-value pair
            dict1[key] = value
    return dict1


# Load every image in the directory
def load_images(image_dir, output_dir):
    try:
        loaded_image_list = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        loaded_image_list.sort()

        processed_image_list = [os.path.join(output_dir, img) for img in os.listdir(output_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        processed_image_list.sort()

        print("Loading images: Completed")

    except Exception as E:
        print(f"Error in load_images: {E}")
        exit(1)

    return loaded_image_list, processed_image_list


# Load the csv data in a dictionary (Rule: the dict key should be in the same format as process_bee_wing arg_list format)
def load_csv(dir):
    # Helper function 1: converts strings in the csv to the correct format
    def format_str(S: str):
        # Check to see if it should be a tuple
        S.strip()
        if '(' and ')' in S:
            # Remove the unnecessary punctuation
            S = S.replace('(', '')
            S = S.replace(')', '')
            S = S.replace(' ', '')

            # Make a temprary list to represent a mutable tuple, and an anchor that marks the comma between the values of the tuple
            temp = list()
            anchor = S.find(',')

            # Add the strings to temp
            temp.append(S[:anchor])
            temp.append(S[anchor + 1:])

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

        elif S.replace('.', '').isdigit():
            return float(S)

        else:
            return S
    
    # Helper function 2: reads every line of csv file and returns the required list and data
    # { Folder_name : { Noise_level : [arg_list] } }
    # ['EX', '46.38573857385739', '15', '19', '99', '(3, 3)', '1', '(3, 3)', '25', '17', '(2, 2)', '(2, 2)', '(5, 5)']
    def row_reader(row: list):
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

        # Return the data
        return data

    # Utilize the helper functions
    try:
        with open(dir, 'r') as csv_file:
            # Read the csv file
            csv_reader = csv.reader(csv_file)

            # Create the data dict\
            data = dict()

            # Iterate the csv file
            for row in csv_reader:
                deep_merge(data, row_reader(row))

            print("Loading CSV: Completed")

            # Return the data
            return data

    except Exception as E:
        print(f"Error in load_csv: {E}")
        exit(1)


# Determine the noise level of a single image
def noise_level_detection(image_dir):
    image = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError
    
    # Use adaptive thresholding for local binarization
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    mean_value = np.mean(thresh)

    return (os.path.basename(image_dir), float(mean_value))


# Process all the images
def process_all_images(input_dir, csv_dir, output_dir):
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
            arg = data[folder_name].get(noise_lvl) or data[folder_name][min(data[folder_name], key = lambda key: abs(key-noise_lvl))]
            global image_counter
            progress_bar(len(desired_images), image_counter, txt="Skeletonization progress: ")
            image_counter += 1
            process_bee_wing(image_dir, arg, output_dir)
        except Exception as E:

            print(f"Error in processing {image_dir}")
            
            global error_counter

            error_counter += 1

            # Log every failed action in log.txt
            with open("log.txt", 'a') as F:
                F.write(f"Error loading image: {os.path.basename(image_dir)}\n\t{E}\n")
                try:
                    F.write(f"\tNoise level: {noise_lvl}\n")
                except Exception as E1:
                    F.write(f"\tLogging \"noise_lvl\" failed: {E1}\n")

                try:
                    F.write(f"\tFolder name: {folder_name}\n")
                except Exception as E2:
                    F.write(f"\tLogging \"folder_name\" failed: {E2}\n")

                try:
                    F.write(f"\tArguments: {arg}\n")
                except Exception as E3:
                    F.write(f"\tLogging \"Arguments\" failed: {E3}\n")

                F.write("\n----------------------------------------------------------------------------------------------------\n")

        
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(process_one_image, desired_images))

    global error_counter
    
    print(f"All images processed with {error_counter} errors")


def main():
    input_dir = r'/home/delta/Documents/original_wings_labeled'
    csv_dir = r'/home/delta/Desktop/CODE/wing2/filter 1.6/tweaks.csv'
    output_dir = r'/home/delta/Documents/modified_wings_labeled'

    process_all_images(input_dir, csv_dir, output_dir)

if __name__ == "__main__":
    main()

# [(PL, 4), (HR, 8), (HU, 1), (MD, 1), (AT, 3), (RO, 2)]