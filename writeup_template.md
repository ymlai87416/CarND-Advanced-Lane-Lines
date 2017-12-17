## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/undistort_output.png "Undistorted"
[image2]: ./writeup_images/undistort_test1.png "Road Transformed"
[image3]: ./writeup_images/color_constancy.png "Color constancy"
[image4]: ./writeup_images/color_detection.png "Binary Example"
[image5]: ./writeup_images/edge_detection.png "Edge Detection"
[image6]: ./writeup_images/road_detection.png "Road Detection"
[image7]: ./writeup_images/perspective_transformation.png "Perspective transformation"
[video8]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 
You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second and the third code cell of the IPython notebook located in "./CarND-Advanced-Lane-Lines-final.ipynb"
The first code cell define the Camera class, which encapsulate all the logic related to camera.

The second code cell show how to use the Camera class to correct an distorted images.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.
 Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each 
 calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy 
 of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the 
 (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients 
using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the 
`cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
The code for this step is contained in the 4th code cell of the IPython notebook located in "./CarND-Advanced-Lane-Lines-final.ipynb"

The RoadFeatureDectector encapsulate all the logic to extract feature from a given image. RoadFeatureDectector reads
parameters given, extract features from the image and create a weighted features array for detection of lane lines.

Below image shows how the road feature dectector works. Left side is the original image, and at the right side, it is 
 the processed feature extraction map. 
![alt text][image4] 

* Red color represent the edges detected by sobel operator
* Blue color represent the white pixels detected
* Green color represent the yellow pixels detected.

In the following section, I am going to explain how I extract the features from a given image.

##### 2.1 Finding white lane lines
In this project, I try to obtains the white pixels by obtaining the intensity level (Y channel in YUV color space) which
 is higher than the 95% of the total image and mark them as white.

On the road, the white lane line may be within the shadow, this will make the pixel represent the white lane line
become a light gray or even a grey. It appears to be white because your brain seeing this gray pixel will try to 
subtract the shadow and think that it is a white.

I have try using CLAHE to restore back the white lane pixel back to white color, but this is not working as expect and  
at the same time overexpose the picture and make the lane detection more difficult. (e.g. make a light gray color region on 
road become a  white region and hence make road detection more difficult)
 
It is not usual for the white pixel detector failed to find out the white lanes by extracting the pixels representing the
white lane, and this is where edge detector helps in detecting the white lane, which is explained later in this section.

##### 2.2 Finding yellow lane lines
To obtain the yellow pixel representing the yellow lane line, I make use of the HSV color space. I specified the lower
 threshold `[20, 20, 60]` and the upper threshold `[80, 255, 255]`
 
The range of hue cover from red to green, not only yellow. It is because yellow objects in dark are likely to be represented by
dark green pixels in the image. This pose a challenge when processing road which has trees and plants near the road side,
 as they are likely to be captured by the yellow pixel detector. To mitigate this effect, I have to employ the road surface
 detector, which is discussed in the following sections.

##### 2.3 Finding edge using sobel operator
Sobel operator is useful for detecting road side, when the color detection failed or when there are no lanes at all.
I apply the Sobel operator across the x axis of the image and take the threshold between `20 - 100`.

Below image shows the case when the lane line is absent and the result of the edge detection.
![alt text][image5]

##### 2.4 Finding road surface using H channel
To filter out noise from the road side, I have to find the road boundary. By removing features beyond the road boundary, 
the lane line detection become more accurate.

To detect road surface. I assume that a pixel near the bottom of the picture is within the road boundary. Flood fill 
algorithm is employed to find the road surface.

There are a lot of noise even on the road surface, so I have used 5 points near the bottom of the picture to find out
the road surface in case flood fill algorithm failed at some points. (e.g. The lane may have a black line in the middle, 
and hence the flood filling on the left side cannot reach the right side.)

H channel of the HSV color space is not sensitive to shadow and I found it suitable to apply flood fill on the H channel.

Here is the result. Red represent the road surface, while the blue dots represent the starting points of the flood fill
algorithm.
![alt text][image6]

##### 2.5 Color constancy
Although color constancy is not used in this project, I would also like to introduce you what color constancy is, and 
what I have found when trying to use it to complete the project.

Color constancy is an example of subjective constancy and a feature of the human color perception system which ensures 
that the perceived color of objects remains relatively constant under varying illumination conditions. In this project, 
it helps the program to identify the color of the pixel irrespective to the ambient light sources.
![alt text][image3]

Although color constancy simply the coding of finding yellow and white color, the time taken to enhancement the image is
too long (reduce to 1.5 frames per second), so I do not use it in this project.

##### 2.6 output

Given the above detectors, I create the final output by the following step
1. Find out the road surface using the gray detector (section 2.4)
2. Find out the yellow pixels (section 2.2)
3. Find out the white pixels (section 2.1)
4. Find out the edges (section 2.3)
5. Mask 2, 3, 4 using the road boundary obtained in step 1.
6. Assign weighting to each type of pixels
    * 5 to yellow pixels detected
    * 2 to white pixels detected
    * 1 to pixel on the edges detected


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is contained in the 2nd code cell of the IPython notebook located in "./CarND-Advanced-Lane-Lines-final.ipynb"

The Camera encapsulate the logic of transforming an image because the transformation is relative to the position of the 
camera.

The code for my perspective transform includes a function called `transform_to_bird_view()`.
The `transform_to_bird_view()` function takes as inputs an image.
I chose the hardcoded the source and destination points in the following manner:

```python
src = np.array([[(189, 720),
                 (590, 450),
                 (689, 450),
                 (1135, 720)
                 ]], dtype=np.float32)

dst = np.array([[(315, 720),
                 (315, 0),
                 (960, 0),
                 (960, 720)
                 ]], dtype=np.float32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 189, 720      | 315, 720    | 
| 590, 450      | 315, 0      |
| 689, 450      | 960, 0      |
| 1135, 720     | 1135, 720   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image
 and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]

To reverse the effect, Camera class provides another function called `transform_to_camera_view()` to transform the 
bird-view image back from camera perspective, and it is used later to annotate the lane line and lane-line pixels on
the image.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this step is contained in the 6th and the 7th code cell of the IPython notebook located in "./CarND-Advanced-Lane-Lines-final.ipynb"

LaneDetector class is used to detect lanes from a given image with the help of Camera class and RoadFeatureDetector class.

VideoLaneDetector class is used to detect lane from a given video, VideoLaneDetector is a subclass of LaneDetector and
it remember the detected lane lines in the previous frame and use them to find the lane in the current frame.


In this project, I have employed 2 ways to identified lane-line pixels and hence the polynomial representing a lane.

* Sliding window searching using convolution
* Lane searching by using the lanes found in previous frames

I am going to discuss them in the coming section.

##### 4.1 Sliding window searching using convolution

The outline of the algorithm is:
1. Find out the starting position of the left and right lane (Discussed in section 4.1.2)
2. Partition the image into layers along the y-axis
3. Try to find the best possible window which contains the pixels of left and right lane lines
4. Using the pixels found in step 3, apply polynomial fitting algorithm to find out the lane lines as a 2nd order polynomial.

###### 4.1.2 Starting points of the lane line
I first take a histogram along all the columns in the lower half of the image like this:
```
import numpy as np
histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
plt.plot(histogram)
```
The result looks like this:

###### 4.1.3 Line tracking algorithm
The algorithm works by cutting the image into `15` layers along the y-axis. At each layer, it apply a convolution to find
out the window (`50` pixels in this project) having the maximum number of "hot" pixels with weighting consider.
A convolution is the summation of the product of two separate signals, in our case the window 
template and the vertical slice of the pixel image.

The algorithm slides the window template across the image from left to right and any overlapping values are summed together, 
creating the convolved signal. The peak of the convolved signal is where there was the highest overlap of pixels and the
 most likely position for the lane marker.
 
One of the short coming of this lane searching algorithm is that it failed when the lane line is close to horizontal 
in the given image, for example, a sharp turn, because the algorithm keeps searching along the y-axis, but the lane is
running along the x-axis.

##### 4.2 Lane searching by using the lanes found in previous frames


To identified the lane line, first we need to find out where the lane line base position is. 

To find out the lane line base position, a histogram of the bottom half of the image is created, then we selected the position
which has the most weighting in left and right part of the histogram.

After the lane line base position is found, the images is divided into 9 or more windows across the vertical direction.
The 1st windows is where the lane line base position is, a rectangle is drawn and the "on" pixels within the windows 
is counted as the lane line pixels. To decide where the next windows is, the pixels within the current windows is 
averaged and the result is used as the center of the next windows.

After all the windows are found and processed, the pixels of the left and right pixels are found, and it is used to find
out the 2nd order polynomial to represent the left and the right lane line. Here is the result.


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The polynomial found for the left and the right lane line is the rescale to the true scale. To calculated the radius 
of the curvature of the lane, we find out the slope at the polynomial near the car. the slope is then used to find out the
radius of the curvature.

For the lane position, as the left and the right lane base position is already known, the position of the middle of the lane
is calcuated and compare it to the center of the picture. The difference is the position of the vehicle with respect to
the center.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

There are 3 videos in the project and here are my final video output

1.  [Project video](https://youtu.be/jYFG3cjl8R0)

2.  [Challenge video](https://youtu.be/7vBXVduiKUc)

3.  [Harder challenge video](https://youtu.be/7HsluwL0tvY)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and
how I might improve it if I were going to pursue this project further.  

#### 2. Future direction
1. Color mask, find out where the road is
2. Lane line data encapsulation
3. Specify what is an error and reject the wrong lane lines calculated.