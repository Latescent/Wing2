import cv2
from PIL import Image
import numpy as np
import time

def preprocess_image(img_gray):
    blurred = cv2.medianBlur(img_gray, 5)

    nlm_denoised = cv2.fastNlMeansDenoising(blurred, None, 30, 7, 21)
    
    _, binary = cv2.threshold(nlm_denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    cv2.imwrite("out1.jpg", binary)

    return opened

def skeletonize(img_bin):
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

    return skel

def postprocess_skeleton(skeleton):
    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
    
    # for i in range(1, num_labels):
    #     if stats[i, cv2.CC_STAT_AREA] < 1000:
    #         skeleton[labels == 1] = 0

    kernel = np.ones((3, 3), np.uint8)
    skeleton = cv2.dilate(skeleton, kernel, iterations=1)

    cv2.imwrite('out3.jpg', skeleton)

    labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(skeleton)
    for label in range(1, labelnum):
        x,y,w,h,size = contours[label]
        if size <= 700:
            skeleton[y:y+h, x:x+w] = 0

    return skeleton

def extract_skeleton(inp, out):
    img_gray = cv2.imread(inp, cv2.IMREAD_GRAYSCALE)
    
    img_bin = preprocess_image(img_gray)
    
    skeleton = skeletonize(img_bin)

    cv2.imwrite('out2.jpg', skeleton)

    skeleton = postprocess_skeleton(skeleton)
    
    cv2.imwrite(out, skeleton)

start = time.time()
extract_skeleton('4.png', 'out.jpg')
end = time.time()

print(end - start)