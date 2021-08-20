import numpy as np
import cv2
import math


def dist(p1,p2):
    return math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )


def transform(pts, matrix):
    p = np.float32([pts[0], pts[1], 1])
    res = np.matmul(matrix, p)
    res = res // res[2]
    return (int(res[0]),int(res[1]))


def faceDet(faces, bodies):
    containsFace = [False]*len(bodies)
    for face in faces:
        if face==[]:
            continue
        for x in range(len(bodies)):
            if intersectsRect(face,bodies[x]):
                containsFace[x]=True
    return containsFace


def intersectsRect(small, big):
    if small[0][0]>=big[0][0] and small[0][0]<=big[1][0] and small[0][1]>=big[0][1] and small[0][1]<=big[1][1]:
        return True
    if small[1][0]>=big[0][0] and small[1][0]<=big[1][0] and small[1][1]>=big[0][1] and small[1][1]<=big[1][1]:
        return True
    return False


def intersectsCircle(c1, c2, r1, r2):
    d = dist(c1,c2)
    if d==0:
        return 100
    return (r1+r2)//d-1


def groupCircles(centers, radii, faceData):
    points=[]
    contFace=[]
    n = len(centers)
    matched = [False]*n
    for x in range(n):
        if not matched[x]:
            count, xCor, yCor, face = 1, centers[x][0],centers[x][1], faceData[x]
            matched[x]=True
            for y in range(x+1,n):
                if not matched[y] and intersectsCircle(centers[x],centers[y],radii[x],radii[y])>1:
                    matched[y]=True
                    count += 1
                    xCor += centers[y][0]
                    yCor += centers[y][1]
                    face = face or faceData[y]
            points.append((xCor//count,yCor//count))
            contFace.append(face)
    return points,contFace


def markPoints(pts, img):
    cv2.circle(img, pts[0], 5, [0, 0, 255], thickness=-1)
    cv2.circle(img, pts[1], 5, [0, 255, 255], thickness=-1)
    cv2.circle(img, pts[2], 5, [0, 255, 0], thickness=-1)
    cv2.circle(img, pts[3], 5, [255, 255, 0], thickness=-1)


pts = [[(178,261),(361,221),(344,356),(540,265)],
       [(274,224),(70,284),(453,258),(247,372)]]

ptsf = [(0,0),(600,0),(0,600),(600,600)]

m = [cv2.getPerspectiveTransform(np.float32(pts[x]),np.float32(ptsf)) for x in range(2)]

cap = [cv2.VideoCapture('E:\Projects\CCTV\outs\\vid1.avi'),
       cv2.VideoCapture('E:\Projects\CCTV\outs\\vid2.avi')]

bodyFiles = [open('E:\Projects\CCTV\outs\\boxes1.txt','r'),
             open('E:\Projects\CCTV\outs\\boxes2.txt', 'r')]

faceFiles = [open('E:\Projects\CCTV\outs\\faces1.txt','r'),
             open('E:\Projects\CCTV\outs\\faces2.txt', 'r')]

res = np.zeros((600, 600, 3))

people = [[],[]]

while True:
    frame = [0]*2
    _,frame[0] = cap[0].read()
    _, frame[1] = cap[1].read()
    res = np.zeros((600, 600, 3))

    markPoints(pts[0],frame[0])
    markPoints(pts[1], frame[1])

    bodies = [[], []]
    faces = [[],[]]

    #Reads bounding boxes for bodies
    line = bodyFiles[0].readline().rstrip()
    if line!='#':
        boxes = line.split(';')
        boxes.pop(-1)
        for box in boxes:
            cords = list(map(int,box.split(',')))
            bodies[0].append([(cords[0],cords[1]),(cords[2],cords[3])])

    line = bodyFiles[1].readline().rstrip()
    if line!='#':
        boxes = line.split(';')
        boxes.pop(-1)
        for box in boxes:
            cords = list(map(int,box.split(',')))
            bodies[1].append([(cords[0],cords[1]),(cords[2],cords[3])])

    #Reads bounding boxes for faces
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

    pos = [[[(body[0][0], body[1][1]), body[1]] for body in camBodies] for camBodies in bodies]

    linePos = [[[transform(pos[x][y][0], m[x]), transform(pos[x][y][1], m[x])] for y in range(len(pos[x]))] for x in
               range(len(pos))]
    radii = [[int(dist(line[0], line[1]) // 2) for line in view] for view in linePos]
    centers = [[((line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2) for line in view] for view in linePos]

    contFace = [faceDet(faces[x], bodies[x]) for x in range(2)]

    for x in range(2):
        for y in range(len(faces[x])):
            cv2.rectangle(frame[x], faces[x][y][0], faces[x][y][1], [0, 255, 255], thickness=2)
        for y in range(len(bodies[x])):
            if contFace[x][y]:
                cv2.rectangle(frame[x], bodies[x][y][0], bodies[x][y][1], [0, 255, 0], thickness=2)
            else:
                cv2.rectangle(frame[x], bodies[x][y][0], bodies[x][y][1], [0, 0, 255], thickness=2)


    points,hasFace = groupCircles(centers[0]+centers[1],radii[0]+radii[1],contFace[0]+contFace[1])

    for x in range(len(points)):
        if hasFace[x]:
            cv2.circle(res, points[x], 5, [0, 255, 0], -1)
            cv2.circle(res, points[x], 30, [0, 255, 0], 2)
        else:
            cv2.circle(res, points[x], 5, [0, 0, 255], -1)
            cv2.circle(res, points[x], 30, [0, 0, 255], 2)

    frame[1]=cv2.flip(frame[1],1)
    cv2.imshow('frame1',frame[0])
    cv2.imshow('frame2', frame[1])
    cv2.imshow("res",res)

    if cv2.waitKey(35)!=-1:
        break

cap[0].release()
cap[1].release()
cv2.destroyAllWindows()