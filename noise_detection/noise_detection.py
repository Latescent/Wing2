import cv2
import shutil

def white_array(img_dir):
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    white_pixels = list()
    height, width = image.shape

    for y in range(height):
        for x in range(width):
            if image[(y, x)] == 255:
                white_pixels.append((x, y))

    return(len(white_pixels))

noisy_list = list()
for i in range(18748):
    if white_array(f"f-{i}.png") > 3500:
        noisy_list.append(f"f-P{i}.png")

for file in noisy_list:
    shutil.copy2(f"./file", './noisy')