import math
import numpy as np

SAMPLE_ROW = 20
SAMPLE_COL = 5
SAMPLE_CNT = SAMPLE_ROW * SAMPLE_COL
SMOOTHNESS = 1000

def GetSamplePoints(rowCnt, colCnt, maxShift):
    points = []
    rowShift = (rowCnt - 2 * maxShift) / (SAMPLE_ROW + 1)
    colShift = (colCnt - 2 * maxShift) / (SAMPLE_COL + 1)
    for i in range(SAMPLE_ROW):
        for j in range(SAMPLE_COL):
            points.append([int(maxShift + i * rowShift), int(maxShift + j * colShift)])
    return points

def GetWeights():
    weights = []
    for i in range(256):
        if (i <= 127):
            weights.append(i / 127)
        else:
            weights.append((255-i) / 127)
    return weights

def LeastSquareSolve(imgs, exposedTime, shifts, sp, w, channel):
    imgCnt = len(imgs)

    ARowCnt = imgCnt * SAMPLE_CNT + 1 + 255
    AColCnt = 256 + SAMPLE_CNT
    A = np.zeros((ARowCnt, AColCnt), dtype=np.float32)
    b = np.zeros(ARowCnt)

    for i in range(imgCnt):
        for j in range(SAMPLE_CNT):
            z = imgs[ i ][ sp[j][0] + shifts[i][0] ][ sp[j][1] + shifts[i][1] ][ channel ]
            row = i * SAMPLE_CNT + j
            A[row][z] = w[z]
            A[row][256 + j] = -w[z]
            b[i] = w[z] * math.log(exposedTime[i])

    row = imgCnt * SAMPLE_CNT
    A[row][127] = 1

    for i in range(1, 256, 1):
        row += 1
        A[row][i-1] = A[row][i+1] = SMOOTHNESS * w[i]
        A[row][i] = -2 * SMOOTHNESS * w[i]

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    g = x[:256]
    lE = x[256:]
    return g, lE

def clamp(input, minValue, maxValue):
    return min(max(input, minValue), maxValue)

def GenerateRadianceMap(imgs, exposedTime, w, gs):
    rowCnt = imgs[0].shape[0]
    colCnt = imgs[0].shape[1]
    radianceMap = np.zeros([rowCnt, colCnt, 3], dtype=np.float32)
    logT = np.log(exposedTime)

    for row in range(rowCnt):
        for col in range(colCnt):
            for channel in range(3):
                numerator = 0
                denominator = 1e-15
                for i in range(len(imgs)):
                    pixel = imgs[i][row][col][channel]
                    numerator += w[pixel] * ( gs[channel][pixel] - logT[i])
                    denominator += w[pixel]
                radianceMap[row, col, channel] = numerator / denominator
    radianceMap = np.exp(radianceMap)
    return radianceMap


def HDR(imgs, exposedTime, shifts):
    sp = GetSamplePoints(imgs[0].shape[0], imgs[0].shape[1], np.max(np.abs(shifts)))
    w = GetWeights()

    gs = []
    for channel in range(3):
        g, _ = LeastSquareSolve(imgs, exposedTime, shifts, sp, w, channel)
        gs.append(g)

    rad = GenerateRadianceMap(imgs, exposedTime, w, gs)
    return gs, rad

