import numpy as np
import cv2
import math
import random

#Version 2, imlements tracking, removes face detection


class Person:
    prevPos = []
    pos = []
    colour = [0,0,0]
    updated=0

    def __init__(self, position):
        self.pos = position
        self.prevPos = position
        self.colour = [random.randint(0,32)*8 for x in range(3)]


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


def groupCircles(centers, radii, faceData):
    points = []
    contFace = []
    n = len(centers)
    matched = [False] * n
    for x in range(n):
        if not matched[x]:
            count, xCor, yCor, face = 1, centers[x][0], centers[x][1], faceData[x]
            matched[x] = True
            for y in range(x + 1, n):
                if not matched[y] and intersectsCircle(centers[x], centers[y], radii[x], radii[y]) > 1:
                    matched[y] = True
                    count += 1
                    xCor += centers[y][0]
                    yCor += centers[y][1]
                    face = face or faceData[y]
            points.append((xCor // count, yCor // count))
            contFace.append(face)
    return points, contFace


def area(rect):
    return (rect[1][0]-rect[0][0])*(rect[1][1]-rect[0][1])


def iouRect(rect1, rect2):
    small = [(max(rect1[0][0],rect2[0][0]),max(rect1[0][1],rect2[0][1])),(min(rect1[1][0],rect2[1][0]),min(rect1[1][1],rect2[1][1]))]
    return area(small)/(area(rect1)+area(rect2)-area(small))


def markPoints(pts, img):
    cv2.circle(img, pts[0], 5, [0, 0, 255], thickness=-1)
    cv2.circle(img, pts[1], 5, [0, 255, 255], thickness=-1)
    cv2.circle(img, pts[2], 5, [0, 255, 0], thickness=-1)
    cv2.circle(img, pts[3], 5, [255, 255, 0], thickness=-1)


def track(people, bodies):
    for view in range(2):
        if bodies[view]==[]:
            continue
        for p in range(len(people[view])):
            people[view][p].updated += 1
            maxVal = 0
            body = -1
            for b in range(len(bodies[view])):
                testVal = iouRect(people[view][p].pos,bodies[view][b])
                if testVal>maxVal:
                    maxVal=testVal
                    body=b

            if body!=-1:
                people[view][p].prevPos = people[view][p].pos
                people[view][p].pos=bodies[view][body]
                people[view][p].updated = 0
                del bodies[view][body]

        for b in range(len(bodies[view])):
            people[view].append(Person(bodies[view][b]))

        for view in range(2):
            for p in range(len(people[view])):
                if p >= len(people[view]):
                    break;
                if people[view][p].updated>10:
                    del people[view][p]

    return people


def lowpass(people, a=0.25):
    for view in range(2):
        for x in range(len(people[view])):
            people[view][x].pos[0] = (int(a * people[view][x].pos[0][0] + (1 - a) * people[view][x].prevPos[0][0]),
                                      int(a * people[view][x].pos[0][1] + (1 - a) * people[view][x].prevPos[0][1]))
            people[view][x].pos[1] = (int(a * people[view][x].pos[1][0] + (1 - a) * people[view][x].prevPos[1][0]),
                                      int(a * people[view][x].pos[1][1] + (1 - a) * people[view][x].prevPos[1][1]))
    return people


pts = [[(178, 261), (361, 221), (344, 356), (540, 265)],
       [(274, 224), (70, 284), (453, 258), (247, 372)]]

ptsf = [(0, 0), (600, 0), (0, 600), (600, 600)]

m = [cv2.getPerspectiveTransform(np.float32(pts[x]), np.float32(ptsf)) for x in range(2)]

cap = [cv2.VideoCapture('/mnt/4A86C22586C2117D/Projects/CCTV/outs/1.avi'),
       cv2.VideoCapture('/mnt/4A86C22586C2117D/Projects/CCTV/outs/2.avi')]

bodyFiles = [open('/mnt/4A86C22586C2117D/Projects/CCTV/outs/boxes1.txt', 'r'),
             open('/mnt/4A86C22586C2117D/Projects/CCTV/outs/boxes2.txt', 'r')]

faceFiles = [open('/mnt/4A86C22586C2117D/Projects/CCTV/outs/faces1.txt', 'r'),
             open('/mnt/4A86C22586C2117D/Projects/CCTV/outs/faces2.txt', 'r')]

res = np.zeros((600, 600, 3))

people = [[], []]
peopleUpdater = [[],[]]
peopleColor = [[],[]]

firstTime = True

while True:
    frame = [0] * 2
    _, frame[0] = cap[0].read()
    _, frame[1] = cap[1].read()
    res = np.zeros((600, 600, 3))

    markPoints(pts[0], frame[0])
    markPoints(pts[1], frame[1])

    bodies = [[], []]
    faces = [[], []]

    # Reads bounding boxes for bodies
    line = bodyFiles[0].readline().rstrip()
    if line != '#':
        boxes = line.split(';')
        boxes.pop(-1)
        for box in boxes:
            cords = list(map(int, box.split(',')))
            bodies[0].append([(cords[0], cords[1]), (cords[2], cords[3])])

    line = bodyFiles[1].readline().rstrip()
    if line != '#':
        boxes = line.split(';')
        boxes.pop(-1)
        for box in boxes:
            cords = list(map(int, box.split(',')))
            bodies[1].append([(cords[0], cords[1]), (cords[2], cords[3])])

    # Reads bounding boxes for faces
    line = faceFiles[0].readline().rstrip()
    if line != '#':
        boxes = line.split(';')
        boxes.pop(-1)
        for box in boxes:
            cords = list(map(int, box.split(',')))
            faces[0].append([(cords[0], cords[1]), (cords[2], cords[3])])

    line = faceFiles[1].readline().rstrip()
    if line != '#':
        boxes = line.split(';')
        boxes.pop(-1)
        for box in boxes:
            cords = list(map(int, box.split(',')))
            faces[1].append([(cords[0], cords[1]), (cords[2], cords[3])])

    people = lowpass(track(people,bodies))

    bodies = [[body.pos for body in view] for view in people]
    pos = [[[(body[0][0], body[1][1]), body[1]] for body in camBodies] for camBodies in bodies]

    linePos = [[[transform(pos[x][y][0], m[x]), transform(pos[x][y][1], m[x])] for y in range(len(pos[x]))] for x in
               range(len(pos))]
    radii = [[int(dist(line[0], line[1]) // 2) for line in view] for view in linePos]
    centers = [[((line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2) for line in view] for view in
                  linePos]

    contFace = [faceDet(faces[x], bodies[x]) for x in range(2)]

    points2, hasFace = groupCircles(centers[0] + centers[1], radii[0] + radii[1], contFace[0] + contFace[1])
    points = centers[0] + centers[1]
    rads = radii[0] + radii[1]
    peeps = people[0] + people[1]

    for x in range(len(points)):
        cv2.circle(res, points[x], 5, [0, 255, 0], -1)
        cv2.circle(res, points[x], rads[x], [x/255 for x in peeps[x].colour], 2)

    for x in range(len(points2)):
        cv2.circle(res, points2[x], 5, [0,255,0], -1)
        cv2.circle(res, points2[x], 15, [0, 255, 0], 2)

    for x in range(2):
        for y in range(len(people[x])):
            cv2.rectangle(frame[x], people[x][y].pos[0], people[x][y].pos[1], people[x][y].colour, 2)

    frame[1] = cv2.flip(frame[1], 1)
    cv2.imshow('frame1', frame[0])
    cv2.imshow('frame2', frame[1])
    cv2.imshow("res", res)

    firstTime = False

    if cv2.waitKey(25) != -1:
        break

cap[0].release()
cap[1].release()
cv2.destroyAllWindows()