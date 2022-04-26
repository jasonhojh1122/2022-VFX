
import feature
import image
import cv2
import sys

if (len(sys.argv) != 2):
    sys.exit("Usage: " + sys.argv[0] + " [Image Set Name]")
IMAGE_SET = sys.argv[1]
PATH = 'data/' + sys.argv[1] + '/'

imgs, focalLength = image.load(PATH)

projected = []
for i in range(len(imgs)):
    projected.append(image.Project(imgs[i], focalLength))

R1, points1 = feature.HarrisDetector(imgs[0])
descriptors1 = feature.GetDescriptors(R1, points1)
projectedPoints1 = feature.ProjectFeaturePoints(points1, imgs[0].shape, focalLength)

cv2.imwrite('result/' + IMAGE_SET + '_feature_points.png', image.drawFeatures(imgs[0], points1) * 255)

stitch = projected[0]
cv2.imwrite('a.png', projected[0]*255)

for i in range(1, len(imgs)):

    R2, points2 = feature.HarrisDetector(imgs[i])
    descriptors2 = feature.GetDescriptors(R2, points2)
    projectedPoints2 = feature.ProjectFeaturePoints(points2, imgs[i].shape, focalLength)

    matches = feature.Match(descriptors1, descriptors2, points1, points2)

    translationImg = image.drawTranslation(imgs[i-1], imgs[i], points1, points2, matches)
    cv2.imwrite('result/' + IMAGE_SET + '_' + str(i-1) + '_to_' + str(i) + '.png', translationImg*255)

    xTranslate, yTranslate = feature.GetTranslate(projectedPoints1, projectedPoints2, matches)

    stitch = image.Stitch(stitch, projected[i], int(xTranslate))

    R1 = R2
    points1 = points2
    descriptors1 = descriptors2
    projectedPoints1 = projectedPoints2

result = image.CropProjected(stitch)
cv2.imwrite('result/' + IMAGE_SET + '.png', result*255)
#cv2.imshow('result', result)
#cv2.waitKey(0)