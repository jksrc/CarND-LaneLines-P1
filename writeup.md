# **Finding Lane Lines on the Road** 

## Writeup

---

The goal of the project was to create a processing pipeline to find lane lines on the road


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"
[blur_image]: ./test_images_output/blur_image.jpg "Blurred Image"
[gray_image]: ./test_images_output/gray_image.jpg "Gray Image"
[edges_image]: ./test_images_output/edges_image.jpg "Edges Image"
[challenge]: test_images_output/challenge.jpg "Challenge Image"
[lines_image]: ./test_images_output/lines_image.jpg "Lines Image"
[lines_default]: ./test_images_output/lines_default.jpg "Default Lines Image"
[RGB_img]: ./test_images_output/RGB_img.jpg "Final Image"
[RGB_img_ex]: ./test_images_output/RGB_img_ex.jpg "Final Image Extrapolated"
[roi_edges_image]: ./test_images_output/roi_edges_image.jpg "ROI Edges Image"
[wimg]: ./test_images_output/wimg.jpg "Weighted Image"
[wimg_ex]: ./test_images_output/wimg_ex.jpg "Weighted Image Extrapolated"
[extrapolation_error]: test_images_output/extrapolation_error.jpg "Extrapolation Error"
[opencv_ui]: test_images_output/opencv_ui.png "OpenCV UI"


---

### Reflection

#### The pipeline 

 The first step was to convert the image to a grayscale image. 
 
    gray_image = grayscale(image)
 
 ![alt text][gray_image]
 
  After that I blurred the image just a little to get rid of some details we do not want. The kernel size of 5 seemed to be 
  appropriate after testing some odd values between 3 and 9.
  
    blur = gaussian_blur(gray, 5)
    
 ![alt text][blur_image]
        
 Next, I used the canny edge detection algorithm to find edges. 
 The parameters for low and high threshold should be in the ration 1:2 or 1:3. So I started of with low values like 7:21
 and still got too many details in the edge-image. Tying 50:150 worked pretty well. Also an automatic approach like described on https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/ using the median of the image and then defining thesholds using a sigma value (default 0.33) gave good results. This approach usually resulted to a threshold ratio of 1:2, e.g. lower=89 and upper=176.
 
    v = np.median(blur)
    sigma = .33
    lower = int(max(0, (1.0 - sigma) * v)) # e.g. 89
    upper = int(min(255, (1.0 + sigma) * v)) # e.  176 
    edges = canny(blur, lower, upper)
    
 ![alt text][edges_image]
   
 In the next step I used some vertices to filter out the region where our lane of interest should be: 
 
    roi_edges = region_of_interest(edges, getVertices(image.shape[1], image.shape[0]))
 
    def getVertices(width, height):
        # x 0
        # 0 0
        x11=0
        y11=height
        # 0 0
        # x 0
        x12=460
        y12=330
        # 0 x
        # 0 0
        x21=510
        y21=330
        # 0 0
        # 0 x
        x22=width
        y22=height
    
        vertices = np.array([[(x11, y11), (x12, y12), (x21,y21), (x22, y22)]],
                            dtype=np.int32)
        return vertices
        
![alt text][roi_edges_image]
  
Now that we have a smaller region I went on with detecting lanes on the edges image. For this purpose I used the 
Probabilistic Hough Line Transform algorithm in OpenCV. This algorithm takes five "adjustment"-parameters as input we have to adjust.
For the first parameter rho (the resolution r parameter) it seems to be common to leave this parameter to 1 pixel. Also se second
resolution parameter theta is commonly set to 1 degree or (PI/180) and is measured in radians. 
   
The bigger challenge was to find good values for the threshold (minimum number of intersections considered to detect a line),
minLinLength (minimumg number of points that can form a line) and maxLineGap (maximum gap between two points to be considered in the same line). 
To get an more efficient approach for empirically finding the right parameters I created a simple generic OpenCV UI with three sliders (and one for taking screenshots).
So with this tool it was much easier to find parameters that allow the algorithm to detect all lines:

![alt text][opencv_ui]
  
   
   
    rho = 1
    theta = np.pi / 180
    threshold = self.uitool.p1()
    min_line_length = self.uitool.p2()
    max_line_gap = self.uitool.p3()
    ...
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
 
 An example result:
 
 ![alt text][lines_image]
 


 
So at end a sample finding is:
 
    threshold = 20
    min_line_length = 12 
    max_line_gap = 10
 
 
 
The resulting "line image" and the originally image where then combined together

    wimg = weighted_img(lines_image, image, .8, 1., 0.0)
           
 ![alt text][wimg]
 

&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;



 ###### Result overview without extrapolation
 
 The following picture shows an overview of the processing of all images with the described settings:
 
  ![alt text][RGB_img]
 

#### Line averaging / extrapolating
 
 
There was also the step to average / extrapolate the lines, so that especially interrupted lines and those that are close together get displayed with a single 
solid line. 

To solve this I used these steps in the draw_lines function:

1. calculate the slope so that we can distinguish between the right and the left line

        for line in lines:
            for x1, y1, x2, y2 in line:
                m = ((y2 - y1) / (x2 - x1))
                if m > 0:
                    rightLines.append(line)
                elif m < 0:
                    leftLines.append(line) 

2. Rearrange all x and y coordinates of the points to conjuncted x and y arrays

        for line in lines:
            for x1, y1, x2, y2 in line:
                m = ((y2 - y1) / (x2 - x1))
                if m > 0:
                    rightLines.append(line)
                elif m < 0:
                    leftLines.append(line)

3. Find coefficients that minimize the squared error on the given x- and y-coordinates array

        z = np.polyfit(x, y, 1)
    
4. Create a one dimensional polynomial

        f = np.poly1d(z)
    
5. Draw line extrapolated

        if type == 'left':
            cv2.line(img, (0, int(f(0))), (img.shape[1], int(f(img.shape[1]))), color, thickness)
        else:
            cv2.line(img, (img.shape[1], int(f(img.shape[1]))), (0, int(f(0))), color, thickness)


&nbsp;
###### Example images with lines extrapolated:

The following images shows an example where the lines got extrapolated:

![alt text][wimg_ex]

&nbsp;

###### Lines extrapolated (all)

And the next image shows an overview of all processed images with extrapolated lines:

![alt text][RGB_img_ex]


### 2. Shortcomings


The video solidYelloLeft.mp4 still contains some outliers when extrapolating the lines. For example, the image below
contains a horizontal line that disrupts the proper line drawing.
 
![alt text][extrapolation_error]

Also my approach of extrapolating using a polynomial function did not create a constant length of lines, so that there 
is a slight up and down movement.

Another problem with the current solution is, that if the ground color changes too much to a brighter value, the lines 
won't be detected anymore. 

And lastly, when executing the challenge video, it was clear that there were still too many details in the image, especially when the
shadow of the trees appeared.


### 3. Suggested improvements

The first thing that was clear when executing the "challenge" video was, that the kernel size of 5 for blurring the image
was too small. E. g. there were too many shadows that caused unwanted edges. An increase to a kernel 
size of 9 for example would help a lot.

Another point for improvement is considering the brightness of the street. There should be a preparation step that
maximizes the contrast of the image, so that the lines come out more clearly.

And finally it would also be beneficial when drawing lines to filter out lines regarding their slope-value,
 so that lines close to horizontal can be ignored.