from fnmatch import translate
import cv2
import numpy as np
import image
import sys

KERNEL_SIZE = 5
KERNEL = (KERNEL_SIZE, KERNEL_SIZE)

def HarrisDetector(img):
    gray = image.Gray(img)
    g = cv2.GaussianBlur(gray, KERNEL, 1, 1)
    IX = cv2.Sobel(g, ddepth=-1, dx=1, dy=0)
    IY = cv2.Sobel(g, ddepth=-1, dx=0, dy=1)
    IXX = IX * IX
    IYY = IY * IY
    IXY = IX * IY
    SXX = cv2.GaussianBlur(IXX, KERNEL, 1, 1)
    SYY = cv2.GaussianBlur(IYY, KERNEL, 1, 1)
    SXY = cv2.GaussianBlur(IXY, KERNEL, 1, 1)
    det = SXX * SYY - SXY * SXY
    trace = SXX + SYY
    R = det + 0.04 * (trace * trace)
    mat = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    maxNeighbor = cv2.dilate(R, mat)
    comp = cv2.compare(R, maxNeighbor, cv2.CMP_EQ )
    comp = comp.astype(np.float32)
    comp /= 255.
    candidates = comp * maxNeighbor

    height = candidates.shape[0]
    width = candidates.shape[1]
    points = []
    for y in range(height):
        for x in range(width):
            if (candidates[y][x] > 0.005):
                points.append((y, x, candidates[y][x]))
    points = sorted(points, key=lambda tup: tup[2], reverse=True)

    return R, points

'''
R = response map
points = [y, x, r]
'''
def GetDescriptors(R, points):
    descriptors = []
    for p in points:
        rect = cv2.getRectSubPix(R, KERNEL, (p[1], p[0]))
        descriptors.append(rect.flatten())
    return descriptors

def Match(descriptors1, descriptors2, points1, points2):
    NStep = 25
    i = 1
    while(True):
        N1 = N2 = i * NStep
        N1 = min(N1, len(points1))
        N2 = min(N2, len(points2))
        matches = []
        for i in range(N1):
            d1 = descriptors1[i]
            distance = []
            for j in range(N2):
                if (abs(points1[i][0] - points2[j][0]) > 20):
                    distance.append( (10000000000, j) )
                elif (points1[i][1] - points2[j][1] < 0):
                    distance.append( (10000000000, j) )
                else:
                    d2 = descriptors2[j]
                    distance.append( (np.linalg.norm(d1-d2), j) )
            distance = sorted(distance, key=lambda tup: tup[0])
            if (distance[0][0] / distance[1][0] <= 0.8):
                matches.append( (i, distance[0][1]) )
        if (len(matches) > 0):
            return matches
        elif (N1 == len(points1) and N2 == len(points2)):
            sys.exit("Failed to match features")
        else:
            i += 1

def GetTranslate(points1, points2, matches):
    translateErr = []
    for i in range(len(matches)):
        err = 0
        m = matches[i]
        xTranslate = points1[m[0]][1] - points2[m[1]][1]
        yTranslate = points1[m[0]][0] - points2[m[1]][0]
        for j in range(len(matches)):
            p1x = points1[matches[j][0]][1]
            p1y = points1[matches[j][0]][0]
            p2x = points2[matches[j][1]][1]
            p2y = points2[matches[j][1]][0]
            xErr = (p1x - xTranslate - p2x)
            yErr = (p1y - yTranslate - p2y)
            err += xErr * xErr + yErr * yErr
        translateErr.append( (err, i) )

    translateErr = sorted(translateErr, key=lambda tup: tup[0])
    m = matches[translateErr[0][1]]
    xTranslate = points1[m[0]][1] - points2[m[1]][1]
    yTranslate = points1[m[0]][0] - points2[m[1]][0]
    print('Translate: ', xTranslate, yTranslate)
    return xTranslate, yTranslate

def ProjectFeaturePoints(points, shape, focal):
    halfH = shape[0] / 2
    halfW = shape[1] / 2
    projected = []
    for p in points:
        theta, h = image.ProjectPoint(p[1], p[0], halfW, halfH, focal)
        projected.append((h, theta, p[2]))
    return projected