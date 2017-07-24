#Here we import libraries whose functions will be called throughout the script.
import cv2
import numpy as np
import argparse
import os
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool

#argparse is a simple way to allow for a command-line interface with the script.
parser = argparse.ArgumentParser(description='High-Throughput Downy Mildew Phenotyping - Finding leaf veins')
parser.add_argument('-ext', type=str, default='jpg',
                    help='Image file extension. Default: jpg')
parser.add_argument('-s', type=int, default=1,
                    help='First image name. Default: 1')
parser.add_argument('-e', type=int, default=1,
                    help='Last image name. If only one image is to be analyzed, set -e to what -s is. Default: 1')
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
parser.add_argument('-linethresh', type=int, default=200,
                    help='The accumulator threshold for the lines in the probabilistic Hough line transform algorithm. Smaller values will mean more lines will be detected. Default: 200')
parser.add_argument('-minline', type=int, default=75,
                    help='The minimum line length that will be dectect with the probabilistic Hough line transform algorithm. Default: 75')
args = parser.parse_args()


#os.path.isdir checks if there is a folder in the current directory (i.e., the directory you are running this script from) called lines.
#If this folder doesn't exist, os.mkdir creates the folder for you.
if (os.path.isdir("lines") == False):
    os.mkdir("lines")

#ThreadPool() tells the program how many CPU cores to use while running the program.
#multiprocessing.cpu_count() obtains the total number of CPU cores that you have. This can be changed to any number,
#but inputting a number greater than the number of CPU cores you have might actually slow down the program.
pool = ThreadPool(multiprocessing.cpu_count())
#my_imgs holds a list of numbers that correspond to image files in the current directory.
#This is used to tell the cores what images you want analyzed and in what order (output order is the same as input order).
my_imgs = list(range(args.s, (args.e + 1)))

def finaldebug(n):
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

    #Here we split out original image in BGR color space into three grayscale images that hold the B, G, and R slices.
    b, g, r = cv2.split(img)

    #The Wallis filter algorithm equalizes the lighting conditions across an image using the local statistics of the image. Given a specified desired mean and standard deviation, the algorithm will break up
    #the image into squares and will adjust the mean and standard deviation to be more like that desired. In our case, we only care about adjusting the standard deviation.
    #This abridged Wallis filter code was implemented with help from Konrad Wenzel, and was originally fully implemented in ImgTool (http://kaydoubleu.de/?page_id=188).

    #We specify how big the sides of the squares will be by the kernel_size parameter. stddev is the desired standard deviation.
    kernel_size = 149
    stddev = 100

    #np.float32 changes the type of b from uint8 to float32. This allows us to have decimals in our calculations. We use the blue slice of the image because it highlights the sporulation phenotype better than other slices.
    b = np.float32(b)
    #cv2.boxFilter convolutes the input image with a kernel matrix filled with ones, the size of which is specified by ksize (149 x 149 kernel in our case). The resulting image pixel values will be means of the pixels in the kernel
    #placed above the corresponding pixels in the original image. ddepth of -1 specifies that mean_mat will be the same type as b (float32).
    mean_mat = cv2.boxFilter(b, ddepth = -1, ksize = (kernel_size, kernel_size))
    #Here we convolute a matrix of squared deviations with a kernel matrix filled with ones, with the resulting matrix, var_mat, holding the variances of the pixels over the kernels.
    var_mat = cv2.boxFilter(cv2.pow(b - mean_mat, 2), ddepth = -1, ksize = (kernel_size, kernel_size))
    #Here we set any negative value in var_mat to 0 as otherwise we cannot take the square root of it. The negative values in the var_mat matrix are very small and are the result of rounding error.
    var_mat[var_mat < 0] = 0
    #stddev_mat gives the standard deviation of the pixels.
    stddev_mat = cv2.sqrt(var_mat)

    #Our Wallis-filtered image can be thought of as being a result of the blue layer being centered and standardized (the (b-mean_mat)/(stddev_mat + 0.00000001) part) and "given" a new standard deviation by
    #the product of the centered and standardized image and stddev. That image then is added to the mean of the original image. This intuitive explanation only differs from reality in the processing of the image
    #through a collection of squares rather than the image as a whole. The addition of 0.00000001 to stddev_mat is used to prevent divison by zero.
    wallis = (b-mean_mat) * (stddev/(stddev_mat + 0.00000001)) + mean_mat
    #Since a grayscale image can only have values between 0 and 255, any values below or above the limit become the limit.
    wallis[wallis > 255] = 255
    wallis[wallis < 0] = 0
    #np.uint8 converts the values in wallis, which are rounded with np.round (Python natively always rounds down when changing variable type), to integers between the values 0 and 255.
    wallis = np.uint8(np.round(wallis))
    #Here we set all the pixels other than 255 to 0.
    ret, wallis = cv2.threshold(wallis, thresh = 254, maxval = 255, type = cv2.THRESH_BINARY)

    # .copy() creates a copy of wallis and stores it in wallispre.
    wallispre = wallis.copy()

    #We convert wallis from a grayscale image to one in RGB color space. The R, G, and B layers become are identical matricies that are equal to the initial grayscale matrix.
    wallis2 = cv2.cvtColor(wallis, code = cv2.COLOR_GRAY2RGB)

    #cv2.HoughLinesP finds lines using the probabilistic Hough transform algorithm. The goal of performing this algorithm is to find leaf veins present on leaf discs and to remove them as otherwise those
    #pixels would be counted towards sporulation quantity. The rho value is the distance resolution of the accumulator in pixels. The theta value of pi/180 radians, which equates to 1 degree, is
    #the angle resolution of the accumulator matrix. The lower the resolutions, the more likely lines will be detected. param2 is the accumulator threshold, i.e., only lines that have votes above this threshold are in
    #the final output. minLineLength is the minimum length in pixels of detected lines. maxLineGap is the maximum gap in pixels between lines that is necessary before the algorithm will link the lines.
    #The output is a 3-dimensional array, where the second dimension is singleton, i.e., it has just one value and therefore serves no purpose, and the first dimension specifies a particular detected line.
    #The third dimension has four values holding x1, y1, x2, and y2, in that order, where the x and y values are that start and end points of the line. As an example, lines[4,0,2] holds the x2 value of the third circle.
    #Remember that in Python, an array's index starts with the value 0 instead of 1.
    lines = cv2.HoughLinesP(wallis, rho = 1, theta = np.pi / 180, threshold = args.linethresh, minLineLength=args.minline,maxLineGap=0)
    #We specify that the loop below will only be run if at least one line was detected.
    if (lines is None) == False:
        #We loop through all the detected lines.
        for i in range(len(lines)):
            #We loop through all the start and end points of a particular line simultaneously.
            for x1, y1, x2, y2 in lines[i]:
                #cv2.line draws a straight line on an input image whose start and end points are pt1 and pt2. Points are specified in the format (x,y).
                #Here we draw the detected lines on wallis2 in green by changing the pixels of the image's green slice to 255.
                cv2.line(wallis2, pt1 = (x1, y1), pt2 = (x2, y2), color = (0, 255, 0), thickness = 15)
                #Here we remove the lines, which are white, from wallis, and we do this by "drawing" the lines in black (the color has one value because wallis is a grayscale image) on the image.
                cv2.line(wallis, pt1 = (x1, y1), pt2 = (x2, y2), color = 0, thickness = 15)

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

    #We loop through all the detected circles.
    for i in range(len(circles[0])):
        #cv2.circle draws a circle on an input image with a particular center point, specified as (x,y), and radius. The circle can be colored using the input image's RGB color space and has a specified thickness in pixels.
        #Here we draw the detected circle's center by setting the radius as 2 pixels, and we use the color purple by changing the pixels of the image's red and blue slices to 255.
        cv2.circle(wallis2, center = (circles[0,i,0], circles[0,i,1]), radius = 2, color = (255, 0, 255), thickness = 5)
        #Here we draw the detected circle.
        cv2.circle(wallis2, center = (circles[0,i,0], circles[0,i,1]), radius = circles[0,i,2], color = (255, 0, 255), thickness = 5)
        #Here we draw the detected circle but with a radius decreased by args.sub. In the getvals.py script, the pixels inside this circle will be counted rather than the original detected circle.
        cv2.circle(wallis2, center = (circles[0,i,0], circles[0,i,1]), radius = (circles[0,i,2] - args.sub), color = (255, 0, 255), thickness = 5)

    #This specifies that a list containing wallis before any modification (wallispre), wallis with the lines to be remove highlighted along with the detected circles (wallis2), and wallis with the detected lines
    #removed will be returned when the finaldebug function is called.
    return [wallispre,wallis2,wallis]

#pool.map calls the finaldebug function using multiple CPU cores with the n variable input of the finaldebug function being a value in my_imgs.
#results holds all the lists of wallispre, wallis2, and wallis in their proper order.
results = pool.map(finaldebug, my_imgs)

#Looping over the index values of the images in results and the initial image index that was specified, we write the images in the results list of lists to the lines folder as files whose name corresponds to the input file name
#with the word img prepended to avoid undesirable sorting of the images in the folder and extension corresponding to the specified extension.
#Images ending in "-1" will be those of wallispre, those ending in "-2" will be those of wallis2, and those ending in "-3" will be those of wallis.
for i, j in zip(range(len(results)), my_imgs):
    cv2.imwrite("lines/img" + str(j) + "-1.%s" % args.ext, results[i][0])
    cv2.imwrite("lines/img" + str(j) + "-2.%s" % args.ext, results[i][1])
    cv2.imwrite("lines/img" + str(j) + "-3.%s" % args.ext, results[i][2])


