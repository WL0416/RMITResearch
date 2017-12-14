import numpy as np
import cv2 as cv
import os

inputpath = "./Maps4Test/"
outputpath = "./GrayScale/"

# loop the target folder to get the original images' information
for root, folders, files in os.walk(inputpath):
    # loop the images
    for file in files:
        image = cv.imread(root+"/"+file)
        # convert image's color into 8-bit gray
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        cv.namedWindow(file, cv.WINDOW_NORMAL)
        cv.imshow(file, image)

        # write out the new accessed images
        cv.imwrite(outputpath+file, grayImage)
        #cv.waitKey(0)
        cv.destroyAllWindows()