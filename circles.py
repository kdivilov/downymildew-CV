#Here we import libraries whose functions will be called throughout the script.
import cv2
import numpy as np
import argparse
import os
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

#argparse is a simple way to allow for a command-line interface with the script.
parser = argparse.ArgumentParser(description='High-Throughput Downy Mildew Phenotyping - Finding leaf discs')
parser.add_argument('-ext', type=str, default='jpg',
                    help='Image file extension. Default: jpg')
parser.add_argument('-s', type=int, default=1,
                    help='First image name. Default: 1')
parser.add_argument('-e', type=int, default=1,
                    help='Last image name. If only one image is to be analyzed, set -e to what -s is. Default: 1')
parser.add_argument('-r', type=int, default=4,
                    help='Number of rows in image. Default: 4')
parser.add_argument('-dist', type=int, default=200,
                    help='Distance in pixels of how far vertically in one direction the program should search for circle centers from the center of a row. Default: 200')
parser.add_argument('-sub', type=int, default=20,
                    help='Value in pixels that will be subtracted from the radius, which is measured also in pixels. This is used to avoid leaf disc outer edges being taken into account in the calculation of disease severity. Default: 20')
parser.add_argument('-l', type=int, default=255,
                    help='Threshold value for L in Lab color space. Default: 255')
parser.add_argument('-a', type=int, default=130,
                    help='Threshold value for a in Lab color space. This threshold is inversed. Default: 130')
parser.add_argument('-b', type=int, default=150,
                    help='Threshold value for b in Lab color space. Default: 150')
parser.add_argument('-can1', type=int, default=5,
                    help='Lower hysteresis parameter in pixels for the Canny filter. Values below this threshold are not kept. Default: 5')
parser.add_argument('-can2', type=int, default=70,
                    help='Upper hysteresis parameter in pixels for the Canny filter. Values above this threshold are kept. Values in between this parameter and the lower parameter only are kept if they are connected by a pixel above this parameter. Default: 70')
parser.add_argument('-minrad', type=int, default=145,
                    help='Minimum circle radius parameter in pixels for the Hough circle transform. Default: 145')
parser.add_argument('-maxrad', type=int, default=210,
                    help='Maximum circle radius parameter in pixels for the Hough circle transform. Default: 210')
parser.add_argument('-cdist', type=int, default=400,
                    help='Minimum distance between the centers of the detected circles. Default: 400')
parser.add_argument('-circlethresh', type=int, default=30,
                    help='The accumulator threshold for the circle centers for the Hough circle transform. Smaller values will mean more circles will be detected. Default: 30')
args = parser.parse_args()


#os.path.isdir checks if there is a folder in the current directory (i.e., the directory you are running this script from) called circles.
#If this folder doesn't exist, os.mkdir creates the folder for you.
if (os.path.isdir("circles") == False):
    os.mkdir("circles")

#ThreadPool() tells the program how many CPU cores to use while running the program.
#multiprocessing.cpu_count() obtains the total number of CPU cores that you have. This can be changed to any number,
#but inputting a number greater than the number of CPU cores you have might actually slow down the program.
pool = ThreadPool(multiprocessing.cpu_count())
#my_imgs holds a list of numbers that correspond to image files in the current directory.
#This is used to tell the cores what images you want analyzed and in what order (output order is the same as input order).
my_imgs = list(range(args.s, (args.e + 1)))

#Here we are defining a function that the cores will iterate over, with n being an image from my_imgs.
def debug(n):
    #cv2.imread reads the (color) image in BGR format. The %s in %s.jpg is changed to n, or the current image the program is analyzing (staring with the first image in my_imgs).
    #The s in %s tells to make n a string (image file names are strings).
    #A color image is loaded into memory as an array of three matricies holding values 0 through 255. Each matrix corresponds to either the blue, green, or red slice of the image.
    #A matrix is a two-dimensional array of numbers arranged in rows and columns. The dimension of the matrix depends on the image resolution.
    img = cv2.imread(filename = '%s.%s' % (n, args.ext))

    #cv2.cvtColor converts a color image from one color space to another. In this case we are converting img from the BGR color space to the LAB color space.
    lab = cv2.cvtColor(img, code = cv2.COLOR_BGR2LAB)
    #cv2.split splits an image into individual slices of a color format. In this case we are splitting an image in LAB color space into three grayscale images that hold the L, A, and B slices.
    #A grayscale image is one that holds values from 0 (black) to 255 (white).
    l,a,b = cv2.split(lab)

    #cv2.threshold takes a grayscale image, a threshold value, a maximum value, and a thresholding type.
    #Using THRESH_BINARY, what cv2.threshold does is it takes all the values in the input grayscale image that are greater than the threshold value and sets them to 255 (white), the maximum value.
    #The rest of the pixels are then set to 0 (black). The output image is stored in thresh_l in this case. ret returns the threshold value.
    ret, thresh_l = cv2.threshold(l, thresh = args.l, maxval = 255, type = cv2.THRESH_BINARY)
    #Using THRESH_BINARY_INV, pixels greater than the threshold value are set to 0 (black), while the rest are set to 255 (white), the maximum value.
    ret, thresh_a = cv2.threshold(a, thresh = args.a, maxval = 255, type = cv2.THRESH_BINARY_INV)
    ret, thresh_b = cv2.threshold(b, thresh = args.b, maxval = 255, type = cv2.THRESH_BINARY)

    #cv2.bitwise_or takes two images (in our case, but it can also take scalars) of the same size and determines which elements of the input matricies are the same and which are different.
    #The same elements are given the value of the two input matricies at that element in the output matrix. The different elements are given the higher value out of the two input matricies at that element in the output matrix.
    mask1 = cv2.bitwise_or(thresh_b, thresh_l)
    #cv2.bitwise_and does the same as above except the different elements are given the lower value out of the two input matricies at that element in the output matrix.
    mask1 = cv2.bitwise_and(mask1, thresh_a)
    #The mask parameter does not alter how cv2.bitwise_and operates, but the final output matrix elements are given values only if the mask parameter input matrix has a value other than 0 in those elements, otherwise
    #the output matrix elements will be 0.
    orig = cv2.bitwise_and(img, img, mask = mask1)

    #.copy() creates a copy of orig and stores it in processed.
    processed = orig.copy()

    #cv2.Canny finds the edges in the input image using the Canny algorithm. The algorithm first finds the edge gradient and direction of each pixel using a Sobel filter with a kernel of the size
    #given by the apertureSize parameter (3x3 kernel in our case). Non-maximum suppression is then applied to set pixels to a value of 0 if they are likely not part of edges. Finally, hysteresis thresholding is applied.
    #Hysteresis thresholding takes two threshold values, threshold1 and threshold2, which correspond to low and high thresholds. If an edge gradient is below the low threshold, that pixel's value is set to 0.
    #If an edge gradient is between the two threshold values, it is kept only if it is connected to a pixel whose value is above the high threshold. cv2.Canny does not perform any Gaussian filter.
    processed = cv2.Canny(image = processed, threshold1 = args.can1, threshold2 = args.can2, apertureSize=3)
    #cv2.GaussianBlur convolutes the input image with a Gaussian kernel, the size of which is specified by ksize (9 x 9 kernel in our case). The sigmaX parameter specifies that the Gaussian filter kernel's standard
    #deviation will be automatically set. This has the effect of blurring the image.
    processed = cv2.GaussianBlur(processed, ksize = (9, 9), sigmaX = 0)

    #cv2.HoughCircles finds circles using the 2-1 Hough Transform algorithm. The goal of this algorithm is to detect all the leaf discs in the image as circles.
    #Canny edge detection is run on the input image prior to running the algorithm, and this cannot be disabled.
    #dp is the inverse ratio of the accumulator resolution to the image resolution. The lower the accumulator resolution, the more votes one will get per circle center.
    #Votes can be thought of as the likelihood of a circle center actually being the center of a circle. Low votes can mean that the circle center might be just the center of noise.
    #param1 corresponds to the high hysteresis threshold above, and the low threshold is param1 divided by 2. param2 is the accumulator threshold, i.e., only circle centers that have votes above this threshold
    #are put in the final output. The minRadius and maxRadius parameters specify the minimum and maximum radii, respectively, in pixels of circles found by the algorithm.
    #The output is a 3-dimensional array, where the first dimension is singleton, i.e., it has just one value and therefore serves no purpose, and the second dimension specifies a particular detected circle.
    #The third dimension has three values holding the x value, y value, and radius of a particular circle, in that order. As an example, circles[0,1,2] holds the radius of the second circle detected. Remember that in Python,
    #an array's index starts with the value 0 instead of 1.
    circles = cv2.HoughCircles(image = processed, method = cv2.HOUGH_GRADIENT, dp = 1, minDist = args.cdist,
                               param1 = 50, param2 = args.circlethresh, minRadius = args.minrad, maxRadius = args.maxrad)

    #np.int32 converts the values in circles, which are rounded with np.round (Python natively always rounds down when changing variable type), to integers between the values -2147483647 and 2147483647.
    circles = np.int32(np.around(circles))

    #Print statements for bug finding
    print('Number of circles found in %s.%s: %s\n' % (n, args.ext,len(circles[0])))

    #We loop through all the detected circles.
    for i in range(len(circles[0])):
        #cv2.circle draws a circle on an input image with a particular center point, specified as (x,y), and radius. The circle can be colored using the input image's RGB color space and has a specified thickness in pixels.
        #Here we draw the detected circle's center by setting the radius as 2 pixels, and we use the color purple by changing the pixels of the image's red and blue slices to 255.
        cv2.circle(orig, center = (circles[0,i,0], circles[0,i,1]), radius = 2, color = (255, 0, 255), thickness = 5)
        #Here we draw the detected circle.
        cv2.circle(orig, center = (circles[0,i,0], circles[0,i,1]), radius = circles[0,i,2], color = (255, 0, 255), thickness = 5)
        #Here we draw the detected circle but with a radius decreased by args.sub. In the getvals.py script, the pixels inside this circle will be counted rather than the original detected circle.
        cv2.circle(orig, center = (circles[0,i,0], circles[0,i,1]), radius = (circles[0,i,2] - args.sub), color = (255, 0, 255), thickness = 5)

    #height and width are the height and width of img.
    height, width = img.shape[:2]

    #We loop over the specified number of rows.
    for i in range(args.r):
        #Similar to cv2.circle, cv2.line draws a straight line on an input image whose start and end points are pt1 and pt2. Points are specified in the format (x,y).
        #Here we draw red horizontal lines, the number of which equal the specified number of rows. The lines' y-axes is given by the formula int((((2 * i) + 1) / (args.r * 2)) * height). args.r is the current
        #row the loop is on, and int() makes whatever is in the parenthesis an integer. The latter is necessary because one cannot draw on a fraction of a pixel.
        cv2.line(orig, pt1 = (0, int((((2 * i) + 1) / (args.r * 2)) * height)), pt2 = (width, int((((2 * i) + 1) / (args.r * 2)) * height)), color = (0, 0, 255), thickness = 5)
        #Here we draw blue vertical lines of length specified by args.dist, the number of which equal the specified number of rows. The lines' x-axes is half the image width.
        #The lines' y-axes is given by the formula int((((2 * i) + 1) / (args.r * 2)) * height + or - args.dist).
        cv2.line(orig, pt1 = (int(0.5 * width), int((((2 * i) + 1) / (args.r * 2)) * height) - args.dist), pt2 = (int(0.5 * width), int((((2 * i) + 1) / (args.r * 2)) * height) + args.dist), color = (255, 0, 0), thickness = 5)

    #This specifies that orig will be returned when the debug function is called.
    return orig

#pool.map calls the crop function using multiple CPU cores with the n variable input of the debug function being a value in my_imgs. results holds all the orig matricies, or images, in their proper order.
results = pool.map(debug, my_imgs)

#Looping over the index values of the images in results and the initial image index that was specified, we write the images in the results list to the circles folder as files whose name corresponds to the input file name and extension corresponding to the specified extension.
for i, j in zip(range(len(results)), my_imgs):
    cv2.imwrite("circles/" + str(j) + ".%s" % args.ext, results[i])
