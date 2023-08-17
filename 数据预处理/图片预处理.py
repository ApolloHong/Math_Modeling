import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

image = cv.imread("C:\\Users\\Apollo\\Pictures\\Screenshots\\屏幕截图_20230110_225345.png")
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
rows, cols, chnl = image_rgb.shape
# 图像下、上、右、左平移
M1 = np.float32([[1, 0, 0], [0, 1, 100]])
image1 = cv.warpAffine(image_rgb, M1, (cols, rows))
M2 = np.float32([[1, 0, 0], [0, 1, -100]])
image2 = cv.warpAffine(image_rgb, M2, (cols, rows))
M3 = np.float32([[1, 0, 100], [0, 1, 0]])
image3 = cv.warpAffine(image_rgb, M3, (cols, rows))
M4 = np.float32([[1, 0, -100], [0, 1, 0]])
image4 = cv.warpAffine(image_rgb, M4, (cols, rows))
# 图像显示
tieles = ["image1", "image2", "image3", "image4"]
images = [image1, image2, image3, image4]
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i])
    plt.xticks([]), plt.yticks([])
    plt.title(tieles[i])
    plt.show()

# 图像的旋转

from PIL import Image

path = ' '
image = Image.open(path)
image.show()
# 左右?平翻转
out = image.transpose(Image.FLIP_LEFT_RIGHT)
# 上下翻转
out = image.transpose(Image.FLIP_TOP_BOTTOM)
# 顺时针旋转90度
out = image.transpose(Image.ROTATE_90)
# 逆时针旋转45度
out = image.rotate(45)
out.save(' ', 'png')

# 图像的缩放

import cv2 as cv

path = ' '
image = cv.imread(path)
out = cv.resize(image, (400, 400))
