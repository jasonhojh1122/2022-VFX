import cv2
from matplotlib import pyplot as plt

import sys
import alignment
import hdr

if (len(sys.argv) != 3):
    print("Usage: ", sys.argv[0], " [Image Set Name]", "[Image Scale Factor]")
IMAGE_SET = sys.argv[1]
PATH = 'data/' + IMAGE_SET + '/'
SCALE_FACTOR = int(sys.argv[2])

def LoadImages():
    f = open(PATH + 'list.txt', 'r')
    lines = f.readlines()
    f.close()
    imgs = []
    exposedTime = []
    for line in lines:
        splitted = line.split(' ')
        fileName = PATH + splitted[0]
        img = cv2.imread(fileName)
        img = cv2.resize(img, [int(img.shape[1]/SCALE_FACTOR), int(img.shape[0]/SCALE_FACTOR),])
        imgs.append(img)
        exposedTime.append(float(splitted[1]))
    return imgs, exposedTime

imgs, exposedTime = LoadImages()
shifts = alignment.Align(imgs)
gs, radianceMap = hdr.HDR(imgs, exposedTime, shifts)

fig = plt.figure()
plt.title('CRF')
plt.plot(gs[0], range(256), 'b', 'o')
plt.plot(gs[1], range(256), 'g', 'o')
plt.plot(gs[2], range(256), 'r', 'o')
plt.xlabel('log exposure')
plt.ylabel('pixel value')
plt.savefig('Result/'+IMAGE_SET+'_CRF.png')

cv2.imwrite('Result/'+IMAGE_SET+'.hdr', radianceMap)
