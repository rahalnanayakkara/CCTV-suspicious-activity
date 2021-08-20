import numpy as np
import cv2
import random
import math

def scale(img, percent):
    width = int(img.shape[1]*percent/100)
    height = int(img.shape[0]*percent/100)
    dim = (width,height)
    return cv2.resize(img,dim, interpolation=cv2.INTER_AREA)


def dist(p1,p2):
    return math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )


def trasform(pts, matrix):
    p = np.float32([pts[0], pts[1], 1])
    res = np.matmul(matrix, p)
    res = res / res[2]
    return (int(res[0]),int(res[1]))


def markPoints(pts, img):
    cv2.circle(img, pts[0], 5, [0, 0, 255], thickness=-1)
    cv2.circle(img, pts[1], 5, [0, 255, 255], thickness=-1)
    cv2.circle(img, pts[2], 5, [0, 255, 0], thickness=-1)
    cv2.circle(img, pts[3], 5, [255, 255, 0], thickness=-1)


def groupPoints(pts,rad=10):
    res =[]
    n = len(pts)
    matched = [False]*n
    for x in range(n):
        if not matched[x]:
            count, xCor, yCor =1,pts[x][0],pts[x][1]
            matched[x]=True
            for y in range(x+1,n):
                if not matched[y] and dist(pts[x],pts[y])<=rad:
                    matched[y]=True
                    count += 1
                    xCor += pts[y][0]
                    yCor += pts[y][1]
            res.append((xCor/count,yCor/count))
    return res


img = [0]*4
img[0] = scale(cv2.imread('E:\Projects\CCTV\set1\img1.jpg',cv2.IMREAD_COLOR),30)
img[1] = scale(cv2.imread('E:\Projects\CCTV\set1\img2.jpg',cv2.IMREAD_COLOR),20)
img[2] = scale(cv2.imread('E:\Projects\CCTV\set1\img3.jpg',cv2.IMREAD_COLOR),30)
img[3] = scale(cv2.imread('E:\Projects\CCTV\set1\img4.jpg',cv2.IMREAD_COLOR),30)

pts = [0]*4
pts[0] = [(446,100),(747,104),(249,590),(907,593)]
pts[1] = [(382,590),(25,596),(280,252),(86,244)]
pts[2] = [(1070,269),(913,616),(361,90),(29,224)]
pts[3] = [(308,595),(76,294),(1044,200),(750,100)]
ptsf = [(0,0),(900,0),(0,300),(900,300)]

bottle = [(534,190),(242,427),(697,255),(555,318)]
cup = [(594,255),(192,370),(527,247),(620,240)]
cone = [(677,440),(152,283),(254,203),(761,154)]

for x in range(4):
    markPoints(pts[x],img[x])

m = [0]*4
result = [0]*4
for x in range(4):
    m[x] = cv2.getPerspectiveTransform(np.float32(pts[x]),np.float32(ptsf))
    result[x] = cv2.warpPerspective(img[x], m[x], (900, 300))


bottleRes = [trasform(bottle[x],m[x]) for x in range(4)]
cupRes = [trasform(cup[x],m[x]) for x in range(4)]
coneRes = [trasform(cone[x],m[x]) for x in range(4)]

resPoints = bottleRes+coneRes+cupRes

gPoints = groupPoints(resPoints)

for p in gPoints:
    cv2.circle(result[0], p, 5, [random.randint(0,255) for x in range(3)], thickness=-1)

cv2.imshow("1.jpg", result[0])

cv2.waitKey(0)
cv2.destroyAllWindows()