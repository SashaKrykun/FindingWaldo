import cv2
import numpy as np

img_rdb = cv2.imread('find.jpeg')
img_gray = cv2.cvtColor(img_rdb, cv2.COLOR_BGR2GRAY)
target_object = cv2.imread('target.png', 0)

width, height = target_object.shape

result = cv2.matchTemplate(img_gray, target_object, cv2.TM_CCORR_NORMED)
threshold = 0.6  # best value what I found for now

loc = np.where(result >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rdb, pt, (pt[0] + width, pt[1] + height), (0, 255, 0), 2)

cv2.imwrite('result.png', img_rdb)
