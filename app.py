from __future__ import division

import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import time

#printing out some stats and plotting


#`cv2.inRange()` for color selection
#`cv2.fillPoly()` for regions selection
#`cv2.line()` to draw lines on an image given endpoints
#`cv2.addWeighted()` to coadd / overlay two images
#`cv2.cvtColor()` to grayscale or change color
#`cv2.imwrite()` to output images to file
#`cv2.bitwise_and()` to apply a mask to an image
from paramUI import ParamUI


class App:
    def __init__(self, videoPath=None, videoOutPath = None):
        self.test ="abc"
        self.start()
        default_threshold = 20 #21 # 13
        default_min_line_length =12 # 16 # 12
        default_max_line_gap = 10#8 # 10
        self.uitool = ParamUI(self, p1=default_threshold, p2=default_min_line_length, p3=default_max_line_gap, p1Title="threshold", p2Title="min_line_length", p3Title="max_line_gap")
        print("starting...")
        self._out = None
        self._extrapolate = True
        if videoPath == None:
            self.uitool.start()
        else:
            #self._videoOutPath = videoOutPath
            self.playVideo(videoPath, videoOutPath)

    def playVideo(self, videoPath, videoOutPath):
        cap = cv2.VideoCapture(videoPath)

        ret = True

        firstCap = cv2.VideoCapture(videoPath)
        success, image = firstCap.read()
        firstCap.release()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if self._extrapolate == True:
            ex = "_ex"
        else:
            ex = ""
        path = videoOutPath.replace(".mp4", "{}.mp4".format(ex))
        out = cv2.VideoWriter(path, fourcc, 10,
                              (image.shape[1], image.shape[0]), True)

        self._out = out
        self._frame_counter = 0

        while (ret):
            ret, frame = cap.read()
            if type(frame) != type(None):
                # print "show.."
                resized = cv2.resize(frame, (960,540), interpolation=cv2.INTER_AREA)
                #self.make_picture("./test_images/challenge.jpg", resized)
                scale_percent = 60  # percent of original size

                conv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = self.update(conv_image)
                #out.write(image)
                cv2.imshow('frame', image)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        sys.exit()


    def start(self):

        result_images = []
        array = []
        array.append('test_images/solidWhiteCurve.jpg')



    def make_picture(self, path, image):
        resized = cv2.resize(image, (284, 216), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, resized)



    def update(self, givenImage=None):
        base_path = "test_images/"
        array=[]
        if type(givenImage) != type(None):
            array.append(givenImage)
        else:
            for image_path in os.listdir(base_path):
                array.append(mpimg.imread(base_path + image_path))


        numpy_v_concat = []
        #print("update...")

        for image in array:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            #print('This image is:', type(image), 'with dimensions:', image.shape)
            gray_image = grayscale(image)
            self.make_picture("./test_images_out/gray_image.jpg", gray_image)
            blur_image = gaussian_blur(gray_image, 9)
            self.make_picture("./test_images_out/blur_image.jpg", blur_image)
            # detect canny params using median
            v = np.median(blur_image)
            sigma = .33
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            print "lower={} upper={}".format(lower, upper)
            edges_image = canny(blur_image, lower, upper)
            self.make_picture("./test_images_out/edges_image.jpg", edges_image)
            #tempedges = canny(blur, self.uitool.p1(), self.uitool.p2())
            roi_edges_image = region_of_interest(edges_image, getVertices(image.shape[1], image.shape[0]))
            self.make_picture("./test_images_out/roi_edges_image.jpg", roi_edges_image)
            rho = 1
            theta = np.pi / 180
            threshold = self.uitool.p1()
            min_line_length = self.uitool.p2()
            max_line_gap = self.uitool.p3()
            # f = fig.add_subplot(rows, columns, i * columns + 1).set_title(
            #     "roh={} theta={:2.4f} threshold={} min_line_length={}  max_line_gap={} i={}".format(rho, theta, threshold,
            #                                                                                         min_line_length,
            #                                                                                         max_line_gap, i))

            if self._extrapolate:
                ex = "_ex"
            else:
                ex = ""

            lines = hough_lines(roi_edges_image, 1, np.pi / 180, threshold, min_line_length, max_line_gap, self._extrapolate)
            lines_image = region_of_interest(lines, getVertices(image.shape[1], image.shape[0]))

            self.make_picture("./test_images_out/lines_image{}.jpg".format(ex), lines_image)
            wimg = weighted_img(lines_image, image, .8, 1., 0.0)
            self.make_picture("./test_images_out/wimg{}.jpg".format(ex), wimg)



            zeroImg = np.zeros_like(image)
            zeroImg[:, :, 2] = roi_edges_image
            #print zeroImg.shape
            zeros = np.zeros((image.shape[1], image.shape[0], 1), dtype=np.uint8)
            zeros2 = np.zeros((image.shape[1], image.shape[0], 1), dtype=np.uint8)
            #zeros_rgb = np.dstack((edges, zeros, zeros2))
            #tmpEdges = cv2.addWeighted(zeros_rgb, 1.0, edges, 1.0, 0.0)
            #print zeros_rgb.shape
            numpy_h_concat = np.concatenate((image, zeroImg), axis=1)
            numpy_h_concat = np.concatenate((numpy_h_concat, wimg), axis=1)
            if len(numpy_v_concat) > 0:
                numpy_v_concat = np.concatenate((numpy_v_concat, numpy_h_concat), axis=0)
            else:
                numpy_v_concat = numpy_h_concat

            if self._out:
                self._frame_counter = self._frame_counter + 1
                cv2.imwrite("./debug/{}.jpg".format(self._frame_counter), numpy_v_concat)
                self._out.write(wimg)
            #return
        resized_image = cv2.resize(numpy_v_concat, (0, 0), None, .35, .35)
        #RGB_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./test_images_out/RGB_img{}.jpg".format(ex), resized_image)
        return resized_image

def getVertices(width, height):
    # x 0
    # 0 0
    x11=0
    y11=height
    # 0 0
    # x 0
    x12=450
    y12=330
    # 0 x
    # 0 0
    x21=520
    y21=330
    # 0 0
    # 0 x
    x22=width
    y22=height

    vertices = np.array([[(x11, y11), (x12, y12), (x21,y21), (x22, y22)]],
                        dtype=np.int32)
    return vertices





def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    #return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_default_lines(img, lines, color, thickness):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    print "should write"
    cv2.imwrite("./test_images_out/lines_default.jpg", img)



def draw_lines(img, lines=[], color=[0, 0, 255], thickness=2, extrapolate=True):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    draw_default_lines(img.copy(), lines, color, thickness)

    if len(lines > 0):

        leftLines = []
        rightLines = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                m = ((y2 - y1) / (x2 - x1))
                #l = math.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)
                if m > 0:
                    rightLines.append(line)
                elif m < 0:
                    #print "m={:2f} len={:2f} p={} (x2-x1)^2 po1={:2f}  po2={:2f}".format(m, l, line, (x2-x1)^2, (y2-y1)^2)
                    leftLines.append(line)



        #print ("left={} right={}".format(len(leftLines), len(rightLines)))

        if len(rightLines) > 0:
            paint_line(img, rightLines, color, thickness, type='right', extrapolate=extrapolate)

        if len(leftLines) > 0:
            paint_line(img, leftLines, color, thickness, extrapolate=extrapolate)

        #plt.show()

    #for line in leftLines:
    #    for x1, y1, x2, y2 in line:
    #        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def paint_line(img, lines, color, thickness, type='left', extrapolate=True):
    x = []
    y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            x += [x1, x2]
            y += [y1, y2]

    z = np.polyfit(x, y, 1)
    f = np.poly1d(z)

    if extrapolate:
        if type == 'left':
            cv2.line(img, (0, int(f(0))), (img.shape[1], int(f(img.shape[1]))), color, thickness)
        else:
            cv2.line(img, (img.shape[1], int(f(img.shape[1]))), (0, int(f(0))), color, thickness)
    else:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, extrapolate):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    draw_lines(line_img, lines, thickness=5, extrapolate=extrapolate)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, a=0.8, b=1., g=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * a + img * b + g
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, g)


def process_image(image):
    "see app class"
    print "ok"
    #return result

#os.listdir("test_images/")

app = App()
#app = App(videoPath="./test_videos/solidWhiteRight.mp4", videoOutPath="./test_videos_output/solidWhiteRight.mp4")
#app = App(videoPath="./test_videos/solidYellowLeft.mp4", videoOutPath="./test_videos_output/solidYellowLeft.mp4")
# app = App(videoPath="./test_videos/challenge.mp4")