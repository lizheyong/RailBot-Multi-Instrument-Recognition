import sys
import cv2
import numpy as np

# Load the grayscale image

img = cv2.imread(r"C:\Users\zheyong\Desktop\digital(1)\chucuo\6_11.jpg", cv2.IMREAD_GRAYSCALE)
print(img.shape)
# Apply Otsu's thresholding method to obtain a binary image
# Here, we want the digits to be white and the background to be black
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 一些yolo裁剪的偏上，导致小数点在下边界上，这时候一腐蚀就没了，可以考虑识”别不出来时候“（或其他判断条件），上下移动框，重新识别
# (竖直方向，水平方向)
if img.shape[0] <= 76:
    # 灰色差不多59、68     膨胀9，4
    ero_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,4))
    dilate_kernel = np.ones((8, 1), np.uint8)
    area_thresh = 8
elif 76< img.shape[0] < 100:
    # 红色差不多77、79
    ero_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 6))
    dilate_kernel = np.ones((5, 1), np.uint8)
    area_thresh = 10
else:
    ero_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    dilate_kernel = np.ones((9, 4), np.uint8)

# Check if the digits are black
# If they are, then the background is white, and we need to invert the image
num_white_pixels = cv2.countNonZero(thresh)
num_black_pixels = thresh.size - cv2.countNonZero(thresh)
if num_white_pixels > num_black_pixels:
    thresh = 255 - thresh

# Erode the binary image using the defined kernel
erosion = cv2.erode(thresh, ero_kernel, iterations=1)
# 进行膨胀操作
dilation = cv2.dilate(erosion, dilate_kernel, iterations=1)

# Show the binary and eroded images
cv2.imshow('Binary Image', thresh)
cv2.imshow('Erosion Image', erosion)
cv2.imshow('Dilation Image', dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find all contours in the eroded image
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Convert the eroded image to grayscale, so we can draw the contour in red
new_gray_img = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)

# Find the smallest contour below 70% of the image height
min_area = None
min_contour = None
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    if erosion.shape[0]*0.98 > y > erosion.shape[0] * 0.7:
        area = cv2.contourArea(contour)
        if area>area_thresh: # 去噪点
            if min_contour is None or area < min_area:
                min_area = area
                min_contour = contour

# If a contour was found, draw it in red and print its x-coordinate
if min_contour is not None:
    x, y, w, h = cv2.boundingRect(min_contour)
    if w/h < 5: # Check if the bounding rectangle is not a square, to avoid mistaking it for a decimal point
        print(f"Contour: x = {x}")
        cv2.drawContours(new_gray_img, [min_contour], -1, (0, 0, 255), 2)
    else:
        print('No decimal point found.')
        sys.exit()

# If no contour was found, print a message and exit the program
else:
    print('No decimal point found.')
    sys.exit()

# Determine the position of the decimal point by counting the number of digits to the right of it
len_num = 4 # Number of digits recognized by the CRNN network (excluding the decimal point)
img_width = img.shape[1]
num_width = img_width/4
for i in range(1, len_num):
    if i*num_width + num_width/2 > x:
        print(f"Decimal point is the {4-i}th digit from the right.")
        break

# Show the image with the contour in red
cv2.imshow('Contours', new_gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()