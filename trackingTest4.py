import kMeanClustering
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def transform(pts, matrix):
    p = np.float32([pts[0], pts[1], 1])
    res = np.matmul(matrix, p)
    res = res // res[2]
    return (int(res[0]), int(res[1]))


def intersectsCircle(c1, c2, r1, r2):
    d = dist(c1, c2)
    if d == 0:
        return 100
    return (r1 + r2) / d


def groupCircles(centers, radii, bodies):
    points = []
    retBodies = []
    marked1 = [False] * len(centers[0])
    marked2 = [False] * len(centers[1])
    for x in range(len(centers[0])):
        for y in range(len(centers[1])):
            if intersectsCircle(centers[0][x], centers[1][y], radii[0][x], radii[1][y]) >= 1:
                points.append(((centers[0][x][0] + centers[1][y][0]) // 2, (centers[0][x][1] + centers[1][y][1]) // 2))
                retBodies.append([bodies[0][x], bodies[1][y]])
                marked1[x] = True
                marked2[y] = True
    for a in range(len(centers[0])):
        if not marked1[a]:
            points.append(centers[0][a])
            retBodies.append([bodies[0][a], []])
    for a in range(len(centers[1])):
        if not marked2[a]:
            points.append(centers[1][a])
            retBodies.append([[], bodies[1][a]])
    return points, retBodies


def markPoints(pts, img):
    cv2.circle(img, pts[0], 5, [0, 0, 255], thickness=-1)
    cv2.circle(img, pts[1], 5, [0, 255, 255], thickness=-1)
    cv2.circle(img, pts[2], 5, [0, 255, 0], thickness=-1)
    cv2.circle(img, pts[3], 5, [255, 255, 0], thickness=-1)


fig = plt.figure()
ax = Axes3D(fig)

pts = [[(178, 261), (361, 221), (344, 356), (540, 265)],
       [(274, 224), (70, 284), (453, 258), (247, 372)]]

xoff = 0
yoff = 0
ptsf = [(0 + xoff, 0 + yoff), (600 + xoff, 0 + yoff), (0 + xoff, 600 + yoff), (600 + xoff, 600 + yoff)]

m = [cv2.getPerspectiveTransform(np.float32(pts[x]), np.float32(ptsf)) for x in range(2)]

ptst = [[transform(pts[x][y], m[x]) for y in range(4)] for x in range(2)]

cap = [cv2.VideoCapture('E:\Projects\CCTV\outs\\vid1.avi'),
       cv2.VideoCapture('E:\Projects\CCTV\outs\\vid2.avi')]

bodyFiles = [open('E:\Projects\CCTV\outs\\boxes1.txt', 'r'),
             open('E:\Projects\CCTV\outs\\boxes2.txt', 'r')]

faceFiles = [open('E:\Projects\CCTV\outs\\faces1.txt', 'r'),
             open('E:\Projects\CCTV\outs\\faces2.txt', 'r')]

count = 0
points = []

while True:
    frame = [0] * 2
    _, frame[0] = cap[0].read()
    _, frame[1] = cap[1].read()

    # refreshes frame
    res = np.zeros((600, 600, 3))

    # marks boundary points
    # markPoints(pts[0], frame[0])
    # markPoints(pts[1], frame[1])
    markPoints(ptst[0], res)
    markPoints(ptst[1], res)

    # body data for each frame
    bodies = [[], []]

    # Reads bounding boxes for bodies
    for x in range(2):
        line = bodyFiles[x].readline().rstrip()
        if line != '#':
            boxes = line.split(';')
            boxes.pop(-1)
            for box in boxes:
                cords = list(map(int, box.split(',')))
                bodies[x].append([(cords[0], cords[1]), (cords[2], cords[3])])

    # position of bottom line of bounding box
    pos = [[[(body[0][0], body[1][1]), body[1]] for body in camBodies] for camBodies in bodies]

    # transformed coordinate of above box
    linePos = [[[transform(pos[x][y][0], m[x]), transform(pos[x][y][1], m[x])] for y in range(len(pos[x]))] for x in
               range(len(pos))]

    # radii and centers of transformed  lines
    radii = [[int(dist(line[0], line[1]) // 2) for line in view] for view in linePos]
    centers = [[((line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2) for line in view] for view in linePos]

    for x in range(2):
        for y in range(len(centers[x])):
            cv2.circle(res, centers[x][y], radii[x][y], [0, 0, 255], thickness=2)

    framePoints, gpBod = groupCircles(centers, radii, bodies)

    frameCenters = []
    frameBodies = []

    i = 0
    for fp in framePoints:
        if 900 > fp[0] > -300 and 900 > fp[1] > -300:
            tp = [fp[0], fp[1], count]
            frameCenters.append(fp)
            frameBodies.append(gpBod[i])
            points.append(tp)
        i += 1

    for p in frameCenters:
        cv2.circle(res, p, 5, [0, 255, 255], thickness=-1)

    count += 1
    if count == 1100:
        break

pp = np.array(points)
clusters = kMeanClustering.Cluster(pp, 6)
for x in range(len(clusters)):
    if len(clusters[x]) > 0:
        pp = np.array(clusters[x])
        print(pp.shape)
        ax.scatter(pp[:, 0], pp[:, 1], pp[:, 2])
plt.show()
