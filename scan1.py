# USAGE
# python scan.py --image images/page.jpg

from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#cl2 = clahe1.apply(gray)
#gray = cv2.GaussianBlur(gray, (5, 5), 0)

gray = cv2.medianBlur(gray,5)
#gray = cv2.bilateralFilter(gray,3,75,75)
th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)


#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#dilated = cv2.dilate(th, kernel, iterations=2)
edged = cv2.Canny(th, 5, 10)

# show the original image and the edge detected image

print("STEP 1: Edge Detection")
cv2.imshow("Thresh Image", th)
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break


try:
	screenCnt

except NameError:
	
	warped = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(warped)
	T = threshold_local(warped , 21, offset = 10, method = "gaussian")
	warped = (warped > T).astype("uint8") * 255
	print("STEP 3: Apply perspective transform")
	cv2.imshow("Original", imutils.resize(orig, height = 650))
	cv2.imshow("Scanned", imutils.resize(warped, height = 650))
	cv2.imshow("HistEqu", imutils.resize(cl1, height = 650))
	cv2.waitKey(0)
	sys.exit(0)

else:

	# show the contour (outline) of the piece of paper
	print("STEP 2: Find contours of paper")
	print(screenCnt)
	cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
	cv2.imshow("Outline", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# apply the four point transform to obtain a top-down
	# view of the original image
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

	# convert the warped image to grayscale, then threshold it
	# to give it the black and white paper effect
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	T = threshold_local(warped, 11, offset = 10, method = "gaussian")
	warped = (warped > T).astype("uint8") * 255

	print("STEP 3: Apply perspective transform")
	cv2.imshow("Original", imutils.resize(orig, height = 650))
	cv2.imshow("Scanned", imutils.resize(warped, height = 650))
	cv2.waitKey(0)