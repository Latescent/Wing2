<<<<<<< HEAD
# Determine the noise level of a single image
import cv2
import numpy as np
import os
import csv

def main():
    with open('/home/delta/Desktop/CODE/wing2/csv_fixer/tweaks.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            image_name = row[0].strip()
            image_path = f'/home/delta/Desktop/CODE/wing2/csv_fixer/original_wings_labeled/{image_name}'
            # fix last line
            if os.path.exists(image_path):
                _, noise_level = noise_level_detection(image_path)
                with open("/home/delta/Desktop/CODE/wing2/csv_fixer/new_tweaks.csv", "a") as file2:
                    writer = csv.writer(file2)
                    writer.writerow([row[1],noise_level,row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12], row[13]])



def write_hedder():
    with open("/home/delta/Desktop/CODE/wing2/csv_fixer/new_tweaks.csv", "w") as file:
        write_hedder = csv.writer(file)
        write_hedder.writerow(["folder_name" ,"noise_level" ,"nlm_h" ,"nlm_tws" ,"nlm_sws" ,"gb_kernel" ,"clahe_cl" ,"clahe_tgs"  ,"thresh_bs" ,"thresh_c" ,"morphx_kernel" ,"kernel_open" ,"kernel_close"])


def noise_level_detection(image_dir="/home/delta/Documents/original_wings_labeled/AT-0001-031-003679-R.dw.png"):
    image = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(True)
        raise ValueError
    
    # Use adaptive thresholding for local binarization
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    mean_value = np.mean(thresh)

    return (os.path.basename(image_dir), float(mean_value))


if __name__ == "__main__":
    main()
=======
#test
#tes
#test3
>>>>>>> 0adcb5cd4d7a9dfcfddc90de618642b8d05c98ba
