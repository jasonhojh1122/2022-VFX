
import cv2
import math
import numpy as np

def load(PATH):
    f = open(PATH + 'list.txt', 'r')
    lines = f.readlines()
    f.close()
    imgs = []
    focal = int(lines[0].strip())
    lines = lines[-len(lines)+1:]
    for line in lines:
        fileName = PATH + line.strip()
        img = cv2.imread(fileName)
        img = img.astype(np.float32) / 255.
        imgs.append(img)
    return imgs, focal

def Project(img, focal):
    projected = np.zeros(img.shape, dtype=np.float32)
    height = img.shape[0]
    width = img.shape[1]
    halfH = height / 2
    halfW = width / 2

    for y in range(height):
        for x in range(width):
            theta, h = ProjectPoint(x, y, halfW, halfH, focal)
            if theta >= 0 or theta < width or h >= 0 or h < height:
                projected[int(h), int(theta), :] = img[y, x, :]

    cropped = Crop(projected)
    return cropped

def ProjectPoint(x, y, halfW, halfH, focal):

    xCentered = x - halfW
    yCentered = y - halfH
    theta = focal * math.atan(xCentered / focal)
    h = focal * yCentered / math.sqrt(xCentered*xCentered + focal*focal)

    theta += halfW
    h += halfH

    return theta, h

def Gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def Crop(img):
    gray = (Gray(img) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minX = minY = 10000000
    maxW = maxH = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        minX = min(x, minX)
        minY = min(y, minY)
        maxW = max(w, maxW)
        maxH = max(h, maxH)

    cropped = img[minY:minY+maxH, minX:minX+maxW]
    return cropped

def Stitch(img1, img2, shift):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    stitch = np.zeros((h1, w1+shift, 3), np.float32)
    stitch[:h1, :w1, :3] = img1
    overlap = w2 - shift
    stitch[:h1, w1:w1+shift, :3] = img2[:h1, overlap:w2, :3]
    for x in range(overlap):
        p = (overlap - x - 1) / (overlap - 1)
        q = 1 - p
        stitch[:h1, w1-overlap+x, :3] = p * img1[:h1, w1-overlap+x, :3] + q * img2[:h1, x, :3]

    return stitch

def CropProjected(img):
    img = Crop(img)
    i = 0
    while(True):
        if img[i, 1, 0] <= 0.0001 and img[i, 1, 1] <= 0.0001 and img[i, 1, 2] <= 0.0001:
            i += 1
        else:
            break
    height = img.shape[0]
    img = img[ i:height-i, :, :]
    return img

def drawTranslation(img1, img2, points1, points2, matches):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img = np.zeros((max(h1, h2), w1+w2, 3), np.float32)
    img[:h1, :w1, :3] = img1
    img[:h2, w1:, :3] = img2
    for m in matches:
        p1 = ( int(points1[m[0]][1]), int(points1[m[0]][0]) )
        p2 = ( int(points2[m[1]][1]) + w1, int(points2[m[1]][0]) )
        cv2.circle(img, p1, 4, (0, 255, 0), 2)
        cv2.circle(img, p2, 4, (0, 255, 0), 2)
        cv2.line(img, p1, p2, (0, 0, 255), 1)
    return img

def drawFeatures(img, points):
    cp = np.copy(img)
    for p in points:
        cv2.circle(cp, (p[1], p[0]), 4, (0, 255, 0), 2)
    return cp