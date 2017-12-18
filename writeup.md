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
[image8]: ./writeup_images/histogram_simple_approach.png "Histogram"
[image9]: ./writeup_images/histogram_simple_fail.png "Histogram fail case"
[image10]: ./writeup_images/guassian_weighting.png "Guassian histogram weighting"
[image11]: ./writeup_images/line_tracking_horizontal.png "Lane detection for shape turn lane"
[image12]: ./writeup_images/line_tracking_hough.png "Detect the average slope of lane to be detected"
[image13]: ./writeup_images/line_tracking_hough_fails.png "Hough fail case"
[image14]: ./writeup_images/lane_search_previous_frame.png "Detect using the detected lane in previous frame"
[image15]: ./writeup_images/test1.jpg "Annotated image with lane line"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 
You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second and the third code cell of the IPython notebook located in "./CarND-Advanced-Lane-Lines-final.ipynb"
The first code cell defines the `Camera` class, which encapsulates all the logic related to the camera.

The third code cell shows how to use the `Camera` class to correct a distorted images.

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

The `RoadFeatureDectector` encapsulates all the logic to create a thresholded binary image from a given image. `RoadFeatureDectector` reads
parameters given, process the image and create the thresholded binary image for lane detection.

The image below shows how `RoadFeatureDetector` works. Left side is the original image, and at the right side, it is 
 the colored thresholded binary image.
 
![alt text][image4] 

* Red color represents the edges detected by Sobel operator
* Blue color represents the white pixels detected
* Green color represents the yellow pixels detected.

In the following section, I am going to explain how I create the thresholded binary image from a given image.

##### 2.1 Finding white lane lines
I try to find out the lower threshold intensity for white pixels by obtaining the intensity level (Y channel in YUV color space) which 
is higher than the 95% of the pixels in the image. It is implemented in the function `__find_white_threshold()`
. The threshold is then used as the input of `__white_color(img, lower)` to find out the white pixels
 

On the road, the white lane line may be covered by the shadow, this will make the pixel represent the white lane line
become a light gray or even a gray. It appears to be white because when your brain seeing this gray pixel will try to 
subtract the shadow and make you think that it is a white lane line.

I have tried to use CLAHE to restore back the white color, but this is not working as expect and 
at the same time introduces noise to the thresholded binary image and makes the lane detection more difficult. (e.g. make a light gray color region on 
the road becomes a  white region and hence make road surface detection return wrong result)
 
It is not unusual for the white pixel detector failing to find out the white pixels representing the
white lane, and this is where the edge detector helps in detecting the white lane, which is explained later in this section.

##### 2.2 Finding yellow lane lines
To obtain the yellow pixels representing the yellow lane line, I make use of the HSV color space. I specified the lower
 threshold `[20, 20, 60]` and the upper threshold `[80, 255, 255]`. The function `__yellow_color()` then takes the image and the lower and 
 upper threshold to search for yellow pixels in the image.
 
As you can see from the lower and upper threshold, the range of hue cover is not only yellow but from red to green. It is because yellow objects under shadow are likely to be represented by
dark green pixels in the image. This poses a challenge when processing road image which has trees and plants near the roadside,
 as pixels of those objects are likely to be extracted out by the yellow pixel detector. To mitigate this effect, I use the road surface
 detector, which is discussed later in this section.

##### 2.3 Finding edge using Sobel operator
Sobel operator is useful in the case when the color detection failed or when there are no lanes at all.
The function `__abs_sobel_thresh` applies the Sobel operator across the x-axis of the image and use the range `20 - 100` as the threshold. 

Below image shows the case when the lane line is absent and the result of the edge detection.

![alt text][image5]

##### 2.4 Finding road surface using H channel
To filter out noise from the roadside, function `__gray_road_detector` is used to find the road boundary. 
By removing pixels beyond the road boundary, the lane line detection becomes more accurate.

To detect road surface. I assume that are some pixels near the bottom of the picture are on the road surface. Flood fill 
algorithm is used to find the road surface.

There are a lot of noise even on the road surface, so I have used 5 points near the bottom of the picture 
in case flood fill algorithm failed at some points. (e.g. The lane may have a black line in the middle, 
and hence the flood filling on the left side cannot reach the right side.)

H channel of the HSV color space is not sensitive to shadow and I found it is suitable to apply flood fill on the H channel.

Here is the result. Red color represents the road surface, while the blue dots represent the starting points of the flood fill
algorithm.

![alt text][image6]

##### 2.5 Color constancy
Although color constancy is not used in this project, I would also like to explain what color constancy is, and 
what I have found when trying to use it to complete the project.

Color constancy is an example of subjective constancy and a feature of the human color perception system which ensures 
that the perceived color of objects remains relatively constant under varying illumination conditions. In this project, 
it helps the program to identify the color of the pixel irrespective of the ambient light sources.

![alt text][image3]

Although color constancy make the code of finding yellow and white color easier to write, the time taken to process the image is
too long (reduce the processing rate to 1.5 frames per second), so I do not use this technique in this project.

##### 2.6 output

Using the above techniques, I create the thresholded binary image by the following steps
1. Find out the road surface (section 2.4)
2. Find out the yellow pixels (section 2.2)
3. Find out the white pixels (section 2.1)
4. Find out the edges (section 2.3)
5. Mask 2, 3, 4 using the road surface boundary obtained in step 1.
6. Assign weighting to each type of pixels in the thresholded binary image
    * 5 to yellow pixels detected
    * 2 to white pixels detected
    * 1 to pixel on the edges detected


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is contained in the 2nd code cell of the IPython notebook located in "./CarND-Advanced-Lane-Lines-final.ipynb"

The `Camera` class encapsulates the logic of transforming an image because the transformation is relative to the position of the 
camera.

The code for my perspective transform is included in a function called `transform_to_bird_view()`.
The `transform_to_bird_view()` function takes as inputs an image.
I hardcoded the source and destination points in the following manner:

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
| 1135, 720     | 960, 720   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image
 and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]

To reverse the effect, the `Camera` class provides another function called `transform_to_camera_view()` to transform the 
bird-view image back from the camera perspective, and it is used later to wrap the detected lane boundary on
the image.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this step is contained in the 6th and the 7th code cell of the IPython notebook located in "./CarND-Advanced-Lane-Lines-final.ipynb"

`LaneDetector` class detects lanes from a given image with the help of `Camera` class and `RoadFeatureDetector` class.

`VideoLaneDetector` class detects lanes from a given video, `VideoLaneDetector` is a subclass of `LaneDetector` and
it stores additional data such as the detected lane in the previous frame and uses them to find the lane in the current frame.

In this project, I use 2 ways to identified lane-line pixels and hence the polynomial representing a lane.

* Sliding window searching using convolution
* Lane searching by using the lanes found in previous frames

I am going to discuss these two approaches later in this section.

##### 4.1 Sliding window searching using convolution

The outline of the algorithm is:
1. Find out the starting position of the left and right lane (Discussed in section 4.1.2)
2. Partition the image into layers along the y-axis
3. Try to find the best possible window which contains the pixels of left and right lane lines
4. Using the pixels found in step 3, apply the polynomial fitting algorithm to find out the lane lines as a 2nd order polynomial.

###### 4.1.2 Starting points of the lane line

I first take a histogram of all the columns in the lower half of the image and the bottom quarter of the image, and then
calculate the starting points of the lane line by a weighted average. (66.6% from the second histogram and 33% from the first histogram)

For images having the lane line as a vertical line, the first histogram works, but for the following image, the first histogram can give 
me the wrong starting point, so I have to factor in the bottom quarter of the image. Below image shows the result of the algorithm.

![alt text][image8]

In function `find_left_and_right_base`
```
import numpy as np
histogram_50 = np.sum(weighted_binary_warped[weighted_binary_warped.shape[0] // 2:, :], axis=0)
histogram_25 = np.sum(weighted_binary_warped[weighted_binary_warped.shape[0] * 3 // 4 :, :], axis=0)

midpoint = np.int(histogram_50.shape[0] // 2)
leftx_base = (np.argmax(histogram_50[:midpoint]) + np.argmax(histogram_25[:midpoint]) *2)//3
rightx_base = (np.argmax(histogram_50[midpoint:]) + np.argmax(histogram_25[midpoint:]) * 2)//3 + midpoint
```

But this will fail sometime, and return a wrong pair of starting position of the left and right lane like following.

![alt text][image9]

To tackle this, the function `find_left_and_right_base_fix_width` take the width of the common road lane in US into 
consideration when finding the starting position of lane lines.

To further improve the detection of starting point using the histogram, I apply gaussian weighting on the thresholded binary image if
I know the starting location of the lane lines in the previous frame.

![alt text][image10]


###### 4.1.3 Lane line pixel searching algorithm

The function `__find_left_and_right_lane_conv` in the class `LaneDetector` does the lane line pixel searching.



The algorithm works by cutting the image into `15` layers along the y-axis. At each layer, it applies a convolution to find
out the window (`50` pixels in this project) having the maximum number of "hot" pixels with weighting consider.
A convolution is the summation of the product of two separate signals, in our case the window 
template and the vertical slice of the pixel image.


The algorithm slides the window template across the image from left to right and any overlapping values are summed together, 
creating the convolved signal. The peak of the convolved signal is where there was the highest overlap of pixels and the
 most likely position for the lane marker.
 
One of the shortcomings of this lane line pixel searching algorithm is that it fails when the lane lines are close to horizontal 
in the thresholded binary image, and this happens when driving around a curve having a small radius because the algorithm 
keeps searching along the y-axis, while the lane lines are running along the x-axis.

To make the lane detection algorithm works again in this situation, I apply the second transformation on the bird view to make
the lane run along the y-axis instead of the x-axis. Below image shows that after the transformation, the lane line pixel searching
algorithm able to locate the lane line pixel without difficulties.

![alt text][image11]

To let the algorithm know when it needs to further transform the thresholded binary image, function `check_vector_helper()` uses 
Hough algorithm to extract all the lines in the image and calculate their median slope. 
If the slope is smaller than 1 or -1, the line tracking algorithm further transforms the thresholded binary image before 
it searches for lane line pixels. The function performing the transformation is `transform_further()` of the `Camera` class (2nd code block)

![alt text][image12]

When there are too many noises in the thresholded binary image, the resulting slope is not as expected, 
like the following image. (Should be > 1 instead of 0.03)

![alt text][image13]


##### 4.2 Lane searching by using the lanes found in previous frames

When I have a highly confident detected lane (see section 4.3). In the next frame of the video I don't need to do a blind search again, 
but instead, I can just search in a margin around the previous line position. For the detail implementation, please
see the function `__find_left_and_right_lane_by_prev_high_conf` in the class  `VideoLaneDetector` at the 7th code cell of
the IPython notebook located in "./CarND-Advanced-Lane-Lines-final.ipynb"

![alt text][image14]

##### 4.3 Sanity check of the lane detected

To confirm that the detected lane is real, I consider the following when implementing the algorithm:

* Checking that there is one lane line starting on the left side, and the other lane starting on the right side of the image.
* Checking that the number of empty windows (windows without enough pixels) is smaller than the given threshold.
* Checking that they are separated by approximately the right distance horizontally
* Checking that they are roughly parallel

If the lane passes all the sanity check, It is stored and is used in the lane detection for the later frames. If the lane is 
good enough (e.g. fulfill at least 2 of the criteria), I will still accept it as the answer, but I will not use the 
detected lane to find out the lane in later frame lane detection because doing so will result in more error. If the lane cannot fulfill more
than 1 criterion, the latest high confidence lane pair is used at the last resort.

The sanity check is implemented in the function `____sanity_check` in  class  `VideoLaneDetector`
while the lane line pair selection is implemented in the function `__detect_lane` in  class  `VideoLaneDetector`

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is contained in the 4th code cell of the IPython notebook located in "./CarND-Advanced-Lane-Lines-final.ipynb"
The class `Lane` encapsulate all the function related to lane e.g. the position of the lane and the curvature of the lane.
It also converts the polynomial to real scale once it knows the pixel to meter ratio.

The polynomials found for the left and the right lane line are rescaled to the true scale. To calculate the radius 
of the curvature of the lane, we find out the slope at the polynomial near the car. the slope is then used to find out the
radius of the curvature. The function calculating the radius are `line_curvature` and `line_curvature_real_scale`


For the lane position, as the left and the right lane starting position is already known, the position of the middle of the lane
is calculated and compare it to the center of the picture. The difference is the position of the vehicle with respect to
the center. The function calculating the radius are `line_x_pos` and `line_base_real_scale`


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `__create_final_image()` of the class `LaneDetector`.  
Here is an example of my result on a test image:

![alt text][image15]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

There are 3 videos in the project and here are my final video output

1.  [Project video](https://youtu.be/9nzhe8n74HE)

2.  [Challenge video](https://youtu.be/xYrTRIrW054)

3.  [Harder challenge video](https://youtu.be/atVWDyXAwI0)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For the feature extraction
1. Sobel operator is more likely to introduce noise to the lane detection algorithm, and the noise sometimes can be 
effectively filtered out by the finding out the road boundary, but sometimes it cannot, like the reflection on the car window shield
2. Detection of color is very difficult when there are not enough ambient light sources. 
3. For the white detector, it works poorly when there are strong light reflections on the road.

For the lane detection:
1. I have improved the lane line pixel searching algorithm to track lane line having slope close to zero. Another problem of lane line pixel searching 
algorithm is that it may merge both lane lines in some case. (e.g. in case of noise). It is because 
the algorithm greedily accepts windows with the maximum number of pixels. 
2. The polyfit algorithm is hard to control. it may return undesired polynomial if there are not enough lane line pixels, or
too many noise pixels.

For the sanity check:
1. Error detection greatly improves the output quality, because it can quantify how good the detected lane is.
and prevent me to use a poorly detected lane for finding the lane in next frames.

For the performance:
1. Performance is around 2-4 frames per second, depends if the algorithm decides to recalculate the lane line using different approaches.
2. In the further, it can be implemented in C++ and use threads to improve the speed of finding lanes. Current performance is too slow and
it cannot be used in real life situation. (> 10 frames per second)

#### 2. Future direction
1. This may be interesting to apply the convolution layer in the NN trained in Behavior cloning project to generate the thresholded binary
image
2. It may be useful to find out the central line of the lane and then find out the lane width, instead of finding
two separate lane lines, because these two lines are likely to have different curvature in noisy or extreme condition.
3. To find out more ways to remove noise when creating thresholded binary image (edge, yellow color ,and white color)