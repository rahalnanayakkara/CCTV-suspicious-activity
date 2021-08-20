import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


boxFile = open('E:\Projects\CCTV\outs\\boxes1.txt', 'r')

cap = cv2.VideoCapture('E:\Projects\CCTV\outs\\vid1.avi')

bodies = []
centers = []
people = [] #centers of people
peopleDet = [] #box details of same people
count = 0
stTemp = []

fig = plt.figure()
ax = Axes3D(fig)

a = 0.8
b = 1

for line in boxFile:
    boxes = line.split(';')
    _, frame = cap.read()
    boxes.pop(-1)
    frameBodies = []
    frameCenters = []
    for box in boxes:
        cords = list(map(int,box.split(',')))
        frameBodies.append([(cords[0],cords[1]),(cords[2],cords[3])])
        center = ((cords[0]+cords[2])//2,cords[3])
        frameCenters.append(center[:])
        stTemp.append([*center[:],count,0,0])
        cv2.circle(frame, center, 5, [0, 255, 0], -1)

    bodies.append(frameBodies)
    centers.append(frameCenters)
    if count == 0:
        x = 0
        for c in centers[0]:
            people.append(stTemp[:])
            peopleDet.append(bodies[0][x])
            print(peopleDet)
            x += 1
    else:
        x = 0
        for c in centers[count]:
            found = False
            for p in people:
                if p[-1][2] >= count-50 and p[-1][2] != count:
                    dist = (c[0]-p[-1][0])**2 + (c[1]-p[-1][1])**2
                    if p[-1][3]!=0 and p[-1][4]!=0 and c[0]!=p[-1][0] and c[1]!=p[-1][1]:
                        dirn = (c[1]-p[-1][1])/(c[0]-p[-1][0])
                        velDir = p[-1][4]/p[-1][3]
                        if dist < 2000 and (dirn / velDir < 0.8 or dirn / velDir > 1.2):
                            p.append([c[0]*b+p[-1][0]*(1-b),c[1]*b+p[-1][1]*(1-b),count,0,0])
                            peopleDet.append(bodies[count][x])
                            found = True
                            break
                    else:
                        if dist < 1000:
                            p.append([*c[:],count,0,0])
                            peopleDet.append(bodies[count][x])
                            found = True
                            break
            if not found:
                people.append([[*c[:],count,0,0]])
                peopleDet.append(bodies[count][x])
            x += 1
    for p in people:
            p[-1][3] = (p[-1][0] - p[-2][0])/(p[-1][2] - p[-2][2])*a + p[-2][3]*(1-a)
            p[-1][4] = (p[-1][1] - p[-2][1])/(p[-1][2] - p[-2][2])*a + p[-2][4]*(1-a)
    cv2.imshow('frame', frame)
    # cv2.waitKey(25)
    count += 1

for p in people:
    pp = np.array(p)
    ax.scatter(pp[:,0], pp[:,1], pp[:,2])
# pt = np.array(people[6])
# ax.scatter(pt[:,0], pt[:,1], pt[:,2])

plt.show()