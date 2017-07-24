#Here we import libraries whose functions will be called throughout the script.
import cv2
import numpy as np
import argparse
import os
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

#argparse is a simple way to allow for a command-line interface with the script.
parser = argparse.ArgumentParser(description='High-Throughput Downy Mildew Phenotyping - Cropping images')
parser.add_argument('-ext', type=str, default='jpg',
                    help='Image file extension. Default: jpg')
parser.add_argument('-s', type=int, default=1,
                    help='First image name. Default: 1')
parser.add_argument('-e', type=int, default=1,
                    help='Last image name. If only one image is to be analyzed, set -e to what -s is. Default: 1')
parser.add_argument('-buff', type=int, default=20,
                    help='Value in pixels that will be added to the sides of the cropped image. This is added as the Hough circle transform is does not perfectly detect leaf discs. Default: 20')
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
parser.add_argument('-minrad', type=int, default=150,
                    help='Minimum circle radius parameter in pixels for the Hough circle transform. Default: 150')
parser.add_argument('-maxrad', type=int, default=210,
                    help='Maximum circle radius parameter in pixels for the Hough circle transform. Default: 210')
parser.add_argument('-cdist', type=int, default=400,
                    help='Minimum distance between the centers of the detected circles. Default: 400')
parser.add_argument('-circlethresh', type=int, default=30,
                    help='The accumulator threshold for the circle centers for the Hough circle transform. Smaller values will mean more circles will be detected. Default: 30')
args = parser.parse_args()


#os.path.isdir checks if there is a folder in the current directory (i.e., the directory you are running this script from) called crop.
#If this folder doesn't exist, os.mkdir creates the folder for you.
if (os.path.isdir("crop") == False):
    os.mkdir("crop")

#ThreadPool() tells the program how many CPU cores to use while running the program.
#multiprocessing.cpu_count() obtains the total number of CPU cores that you have. This can be changed to any number,
#but inputting a number greater than the number of CPU cores you have might actually slow down the program.
pool = ThreadPool(multiprocessing.cpu_count())
#my_imgs holds a list of numbers that correspond to image files in the current directory.
#This is used to tell the cores what images you want analyzed and in what order (output order is the same as input order).
my_imgs = list(range(args.s, (args.e + 1)))

#Here we are defining a function that the cores will iterate over, with n being an image from my_imgs.
def crop(n):
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
    #The third dimension has three values, 0, 1, 2, and holds the x value, y value, and radius of a particular circle. As an example, circles[0,1,2] holds the radius of the second circle detected. Remember that in Python,
    #an array's index starts with the value 0 instead of 1.
    circles = cv2.HoughCircles(image = processed, method = cv2.HOUGH_GRADIENT, dp = 1, minDist = args.cdist,
                               param1 = 50, param2 = args.circlethresh, minRadius = args.minrad, maxRadius = args.maxrad)

    #np.int32 converts the values in circles, which are rounded with np.round (Python natively always rounds down when changing variable type), to integers between the values -2147483647 and 2147483647.
    circles = np.int32(np.around(circles))

    #height and width are the height and width of img.
    height, width = img.shape[:2]

    #hold is initialized as an empty list.
    hold = []
    #We loop over all the circles detected with the Hough Transform algorithm.
    for i in range(len(circles[0])):
        #If any of the detected circles' x values or y values are outside of our image dimensions, we append the index of that circle in the circles variable to the hold list.
        if (circles[0,i,0] - circles[0,i,2] < 0) or (circles[0,i,0] + circles[0,i,2] > width) or (
                circles[0,i,1] - circles[0,i,2] < 0) or (circles[0,i,1] + circles[0,i,2] > height):
            hold.append(i)

    #np.delete deletes rows (specified by axis = 0) from circles[0]. The rows it will delete are those held in hold. The new circles variable will be 2-dimensional instead of 3-dimensional, with the first unnecessary dimension lost.
    circles = np.delete(circles[0], hold, axis = 0)

    #croptop is a list that holds the y values of all the circles subtracted by the radius of the respective circles.
    croptop = []
    #cropbot is a list that holds the sum of y values of all the circles and the radius of the respective circles.
    cropbot = []
    #cropright is a list that holds the sum of x values of all the circles and the radius of the respective circles.
    cropright = []
    #cropleft is a list that holds the x values of all the circles subtracted by the radius of the respective circles.
    cropleft = []

    #We loop over all the detected circles and fill the lists created above.
    for i in range(len(circles)):
        croptop.append(circles[i,1] - circles[i,2])
        cropbot.append(circles[i,1] + circles[i,2])
        cropright.append(circles[i,0] + circles[i,2])
        cropleft.append(circles[i,0] - circles[i,2])

    #crop_img will be a cropped version of the original image where the new borders correspond to the circle edges' x and y values closest to the border. args.buff is an optional buffer that could be placed
    #around the image. This is done as the Hough circle transform algorithm is not perfect in detecting leaf discs. This allows the cropped image to wholly retain the leaf discs on the edges of the image.
    #max is used here as the original image matrix cannot have negative rows or columns.
    crop_img = img[max(0, np.amin(croptop) - args.buff):max(0, np.amax(cropbot) + args.buff),
                   max(0, np.amin(cropleft) - args.buff):max(0, np.amax(cropright) + args.buff)]

    #This specifies that crop_img will be returned when the crop function is called.
    return crop_img

#pool.map calls the crop function using multiple CPU cores with the n variable input of the crop function being a value in my_imgs. results holds all the crop_img matricies, or images, in their proper order.
results = pool.map(crop, my_imgs)

#Looping over the index values of the images in results and the initial image index that was specified, we write the images in the results list to the crop folder as files whose name corresponds to the input file name and extension corresponding to the specified extension.
for i, j in zip(range(len(results)), my_imgs):
    cv2.imwrite("crop/" + str(j) + ".%s" % args.ext, results[i])
