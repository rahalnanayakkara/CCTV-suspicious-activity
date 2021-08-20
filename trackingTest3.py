import random
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


def faceDet(faces, bodies):
    containsFace = [False] * len(bodies)
    for face in faces:
        if face == []:
            continue
        for x in range(len(bodies)):
            if intersectsRect(face, bodies[x]):
                containsFace[x] = True
    return containsFace


def intersectsRect(small, big):
    if small[0][0] >= big[0][0] and small[0][0] <= big[1][0] and small[0][1] >= big[0][1] and small[0][1] <= big[1][1]:
        return True
    if small[1][0] >= big[0][0] and small[1][0] <= big[1][0] and small[1][1] >= big[0][1] and small[1][1] <= big[1][1]:
        return True
    return False


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


# lowpass constatns
a = 0.6
b = 0.6

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

frame_width = [int(cap[0].get(3)), int(cap[1].get(3))]
frame_height = [int(cap[0].get(4)), int(cap[1].get(4))]

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = [cv2.VideoWriter('E:\Projects\CCTV\outs\\vidout1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                       (frame_width[0], frame_height[1])),
       cv2.VideoWriter('E:\Projects\CCTV\outs\\vidout2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                       (frame_width[0], frame_height[1]))]

outres = cv2.VideoWriter('E:\Projects\CCTV\outs\\vidout3.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                         (600, 600))

# format := [xCor, yCor, time, velX, velY]
people = []

# [(x1,y1),(x2,y2)] view1 , [(x1,y1),(x2,y2)] view1, [B,G,R] color]
peopleBox = []

points = []

# res = np.zeros((600, 600, 3))

count = 0

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

    for i in range(len(framePoints)):
        if 900 > framePoints[i][0] > -300 and 900 > framePoints[i][1] > -300:
            tp = [framePoints[i][0], framePoints[i][1], count]
            frameCenters.append(framePoints[i])
            frameBodies.append(gpBod[i])
            points.append(tp)

    for p in frameCenters:
        cv2.circle(res, p, 5, [0, 255, 255], thickness=-1)

    for i in range(len(frameCenters)):
        c = frameCenters[i]
        fb = frameBodies[i]
        found = False
        for j in range(len(people)):
            p = people[j]
            pb = peopleBox[j]
            if p[-1][2] >= count - 50 and p[-1][2] != count:
                distance = (c[0] - p[-1][0]) ** 2 + (c[1] - p[-1][1]) ** 2
                if p[-1][3] != 0 and p[-1][4] != 0 and c[0] != p[-1][0] and c[1] != p[-1][1]:
                    dirn = (c[1] - p[-1][1]) / (c[0] - p[-1][0])
                    velDir = p[-1][4] / p[-1][3]
                    if distance < 10000 and (dirn / velDir < 0.8 or dirn / velDir > 1.2):
                        p.append([c[0] * b + p[-1][0] * (1 - b), c[1] * b + p[-1][1] * (1 - b), count, 0, 0])
                        pb.append([*fb, pb[-1][2]])
                        found = True
                        break
                else:
                    if distance < 3000:
                        p.append([*c[:], count, 0, 0])
                        pb.append([*fb, pb[-1][2]])
                        found = True
                        break
        if not found:
            people.append([[*c[:], count, 0, 0]])
            peopleBox.append([[*fb, [random.randint(100, 256), random.randint(100, 256), random.randint(100, 256)]]])

    # calculates velocities
    for p in people:
        if len(p) > 1:
            p[-1][3] = (p[-1][0] - p[-2][0]) / (p[-1][2] - p[-2][2]) * a + p[-2][3] * (1 - a)
            p[-1][4] = (p[-1][1] - p[-2][1]) / (p[-1][2] - p[-2][2]) * a + p[-2][4] * (1 - a)

    for i in range(len(people)):
        if len(people[i]) > 10 and people[i][-1][2] >= count - 5:
            if peopleBox[i][-1][0] != []:
                cv2.rectangle(frame[0], peopleBox[i][-1][0][0], peopleBox[i][-1][0][1], peopleBox[i][-1][2],
                              thickness=2)
            if peopleBox[i][-1][1] != []:
                cv2.rectangle(frame[1], peopleBox[i][-1][1][0], peopleBox[i][-1][1][1], peopleBox[i][-1][2],
                              thickness=2)

    frame[1] = cv2.flip(frame[1], 1)
    cv2.imshow('frame1', frame[0])
    cv2.imshow('frame2', frame[1])
    cv2.imshow("res", res)

    for i in range(2):
        out[i].write(frame[i])
    outres.write(res)

    if cv2.waitKey(1) != -1:
        break

    count += 1
    if count == 1100:
        break

for p in people:
    if len(p)>20:
        pp = np.array(p)
        ax.scatter(pp[:, 0], pp[:, 1], pp[:, 2])
plt.show()

outres.release()
cap[0].release()
cap[1].release()
out[0].release()
out[1].release()
cv2.destroyAllWindows()
