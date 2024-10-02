from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import cv2
import numpy as np
import time


def adjust_brightness(inp, out):
    img = Image.open(inp)
    brightness_range = [175, 185]

    def get_brightness():
        value_list = []
        for i in range(img.width):
            for j in range(img.height):
                value_list.append(img.getpixel((i, j)))

        avg = sum(value_list) / len(value_list)

        return int(avg)
    
    while True:
        avg = get_brightness()
        if brightness_range[0] <= avg <= brightness_range[1]:
            break
        
        distance = 180 - avg
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance((100 + distance) / 100)

    img.save(out)


def gray_filter(inp, out):
    img = cv2.imread(inp)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(out, gray)


def median_filter(inp, out):
    img_noisy = cv2.imread(inp, 0)
    img_new = cv2.medianBlur(img_noisy, 3)
    cv2.imwrite(out, img_new)


def binary_filter(inp, out):
    img = cv2.imread(inp)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    cv2.imwrite(out, thresh)


def invert_filter(inp, out):
    img = Image.open(inp)
    inv_img = ImageChops.invert(img)
    inv_img.save(out)


def gaussian_filter(inp, out):
    image = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite(out, image)


def preprocess_image(img_bin):
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(img_bin, kernel, iterations=1)
    return dilated

def skeleton_filter(inp, out):
    img_bin = cv2.imread(inp, 0)

    img_bin = preprocess_image(img_bin)

    size = np.size(img_bin)
    skel = np.zeros(img_bin.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img_bin, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img_bin, temp)
        skel = cv2.bitwise_or(skel, temp)
        img_bin = eroded.copy()

        zeros = size - cv2.countNonZero(img_bin)
        if zeros == size:
            done = True

    skel_inv = cv2.bitwise_not(skel)

    cv2.imwrite(out, skel_inv)


def crop_wing(inp, out):
    image = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]
        cropped_pil_image = Image.fromarray(cropped_image)

        cropped_pil_image.save(out)
    else:
        raise ValueError
        print("No significant object found.")


def noise_removal(inp, out):
    gray_filter(inp, 'out1.jpg')
    adjust_brightness('out1.jpg', 'out2.jpg')

    median_filter('out2.jpg', 'out3.jpg')
    # average_filter('out3.jpg', 'out4.jpg')
    gaussian_filter('out3.jpg', 'out5.jpg')
    invert_filter('out5.jpg', 'out6.jpg')
    binary_filter('out6.jpg', 'out7.jpg')
    # median_filter('out7.jpg', out)
    # average_filter(out, out)
    # gaussian_filter(out, out)
    # binary_filter(out, 'out7.jpg')
    skeleton_filter('out7.jpg', 'out8.jpg')
    invert_filter('out8.jpg', 'out9.jpg')
    crop_wing('out9.jpg', 'out10.jpg')


start_time = time.time()

noise_removal('4.png', 'out.jpg')

end_time = time.time()

print(end_time - start_time)