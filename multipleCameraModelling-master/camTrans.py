import matplotlib.pyplot as plt
import cv2
import numpy as np


def trasform(pts, matrix):
    p = np.float32([pts[0], pts[1], 1])
    res = np.matmul(matrix, p)
    res = res / res[2]
    return (int(res[0]),int(res[1]))


img = cv2.imread('/mnt/4A86C22586C2117D/Projects/CCTV/test2.jpg',cv2.IMREAD_COLOR)

print('Original Dimensions : ', img.shape)

scale_percent = 100  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

print('Resized Dimensions : ', resized.shape)

cv2.circle(resized,(235,474),5,[0,0,255],thickness=-1)
cv2.circle(resized,(993,477),5,[0,0,255],thickness=-1)
cv2.circle(resized,(95,625),5,[0,0,255],thickness=-1)
cv2.circle(resized,(1120,629),5,[0,0,255],thickness=-1)

pts1 = np.float32([(235,474),(993,477),(95,625),(1120,629)])
pts2 = np.float32([(0,0),(900,0),(0,300),(900,300)])

matrix = cv2.getPerspectiveTransform(pts1,pts2)

result = cv2.warpPerspective(resized, matrix, (900,300))

p = (389,557)
cv2.circle(resized,p,5,[0,255,0],thickness=-1)
cv2.circle(result,trasform(p,matrix),5,[0,255,0],thickness=-1)

cv2.imshow("Resized image", resized)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()