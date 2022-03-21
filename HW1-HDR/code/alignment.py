import numpy as np
import cv2

MAX_MTB_EXP = 3
BITMAP_THRESH_OFFSET = 10

def Align(imgs) -> list[int]:

    centerGray = cv2.cvtColor(imgs[int(len(imgs)/2)], cv2.COLOR_BGR2GRAY)
    centerBitMaps = []
    masks = []
    for exp in range(0, MAX_MTB_EXP+1, 1):
        cols = int(centerGray.shape[0] / (1 << exp))
        rows = int(centerGray.shape[1] / (1 << exp))
        resized = cv2.resize(centerGray, (rows, cols))
        mid = np.median(resized)
        masks.append(cv2.inRange(resized, mid - BITMAP_THRESH_OFFSET, mid + BITMAP_THRESH_OFFSET))
        centerBitMaps.append(CreateBitMap(resized))
    # cv2.imshow('mask', masks[0])
    # cv2.waitKey(0)
    # cv2.imshow('bit', centerBitMaps[0])
    # cv2.waitKey(0)

    shifts = []

    for i in range(0, len(imgs), 1):
        imgGray = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
        rowOffset = colOffset = 0
        for exp in range(MAX_MTB_EXP, -1, -1):
            cols = int(imgGray.shape[0] / (1 << exp))
            rows = int(imgGray.shape[1] / (1 << exp))
            resized = cv2.resize(imgGray, (rows, cols))
            imgBitMap = CreateBitMap(resized)
            r, c = MTBAlign(centerBitMaps[exp], masks[exp], imgBitMap, rowOffset, colOffset)
            rowOffset = rowOffset * 2 + r
            colOffset = colOffset * 2 + c
        shifts.append([rowOffset, colOffset])

    return shifts

def MTBAlign(centerBitMap, mask, imgBitMap, rowOffset, colOffset):
    minDiff = centerBitMap.shape[0] * centerBitMap.shape[1]
    r = c = 0
    rows, cols = centerBitMap.shape[:2]
    for rowShift in range(-1, 2, 1):
        for colShift in range(-1 ,2, 1):
            m = np.float32([[1, 0, rowShift+rowOffset], [0, 1, colShift+colOffset]])
            shifted = cv2.warpAffine(imgBitMap, m, (cols, rows))
            diff = CompareBitMap(centerBitMap, shifted, mask)
            if (diff < minDiff):
                minDiff = diff
                r = rowShift
                c = colShift
    return (r, c)

def CreateBitMap(img):
    median = np.median(img)
    _, bitmap = cv2.threshold(img, median, 255, cv2.THRESH_BINARY)
    return bitmap

def CompareBitMap(map1, map2, mask):
    ret = cv2.bitwise_xor(map1, map2)
    ret = cv2.bitwise_and(ret, mask)
    return cv2.countNonZero(ret)