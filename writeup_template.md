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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 
You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb"
 (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.
 Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients 
using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

#### 2.1 Improving the color of the image

##### 2.1.1 Color constancy
Color constancy is an example of subjective constancy and a feature of the human color perception system which ensures 
that the perceived color of objects remains relatively constant under varying illumination conditions. In this project, 
it helps the program to identify the color of the pixel irrespective to the shadow on the road.

##### 2.1.1 White lane line
On the road, the white lane line may be within the shadow, this will make the pixel represent the white lane line
become a light gray or even a grey. It appears to be white because your brain seeing this gray pixel will try to 
subtract the shadow and think that it is a white.

To enhance the image so that the white lane, using CLAHE can improve the image so that pixel can have enough intensity 
to pass though the threshold set by the program. Sometime, even CLAHE cannot give a satisfying image, retinex algorithm 
can be used. In this project, If the program detected the image quality is not enough, it will switch to automated MSRCR
as a final resort to improve image quality

##### 2.1.2 Yellow lane line
Same logic according to the white lane line, but using CLAHE to restore the color of yellow line is undesirable in dark 
images, which result in a washed-out yellow. using automated MSRCR can return a yellow lane.

##### 2.2.2 Creating a binary image

There are 2 steps to create a binary image.

First, we create a color mask to mask out all the irrelevant area in the image.

1. A color filter to search for yellow color in LAB B channel
2. A color filter to search for grey - white color in RGB color space
3. A color filter to filter out black color in RGB color space

Then I apply the following on the image to create the binary image.

1. $$Sobel_X$$ kernel to detect the change in the horizontal direction within the image
2. use HSL color space to search for color having S channel within 170-255
3. use RGB color space to search for light gray to white

Here's an example of my output for this step.  (note: this is not actually from one of the test images)

Color mask

![alt text][image3]

Binary image

![alt text][image3]

Resulting image
=======
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 
# through # in `another_file.py`).  Here's an example of my output for this step. 
(note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the 
file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook). 
The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. 
I chose the hardcode the source and destination points in the following manner:


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

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

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