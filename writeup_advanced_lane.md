

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

[figure1]: ./figures/before_after_cam_cal.png "Camera Calibration"
[figure2]: ./figures/undistort_image.png "Undistorted"
[figure3]: ./figures/lane_reveal_latest.png "Binary Image"
[figure4]: ./figures/source_destination.png "Perspective Transform"
[figure5]: ./figures/convolution_sliding_window.png "Sliding Window"
[figure6]: ./figures/example_output.png "Example Output"
[figure7]: ./figures/system_reset_2ndCement.png "System Reset Example"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
***Here I will consider the rubric points individually and describe how I addressed each point in my implementation. ***

***Writeup / README ***

You're reading it!

***Camera Calibration***

****1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.****

The code for this step is contained in the calibrate_camera.py module. The function is camera_calibrate(). It will save the camera matrix and distortion values in a pickle file for later use in the pipeline.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][figure1]


***Pipeline (single images)***

****1. Provide an example of a distortion-corrected image.****
The camera_calibrate() function writes the camera matrix and distortion to a pickle file. To undistort an image you use the following code:


`dist_pickle = pickle.load( open('camera_cal/wide_dist_pickle.p','rb'))`


 `dst = cv2.undistort(img, dist_pickle['mtx'],dist_pickle['dist'], None,        dist_pickle['mtx'])`

Below is example image before and after undistort:
![alt text][figure2]
****2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.****

I used a combination of color and gradient thresholds to generate a binary image. The code is in the  `reveal_lanes_utils.py module`. The main function is `reveal_lanes()` located between lines 172 and 213. I experimented with several different techniques and there are several functions in `reveal_lanes_utils.py` that were experimented with and not used but left for further exploration. I ended up using only a combination of three masks. A yellow mask, a highlights mask, and a sobel threshold on the l channel in x direction from an HLS color transform. An example of the individual masks and final combination from the `reveal_lanes()` function is shown below:

![alt text][figure3]

****3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.****

The code for my perspective transforms is in perspective_transform.py between lines 9 and 32. The first function is `init_perspective_transforms()` and it calls `get_src_dst()` which has the source and destination points hard coded. The function `init_perspective_transforms()` returns both the forward and inverse transform variable M. Then I use `cv2.warpPerspective()` with the forward M value for forward transform and with the inverse M value for inverse.

I used the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 230, 703      | 480, 720        |
| 573, 466      | 480, 0      |
| 709, 466      | 830, 0      |
| 1080, 703     | 830, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image as shown below:

![alt text][figure4]

****4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?****

The functions for identifying the lane pixels are in `sliding_window.py`. I modified the sliding window convolution method given in the lecture substantially. I found and posted solutions to two errors in the original lecture code (issue 543). The main two pieces of code are `find_lane_centroids()` (lines 57-150), and `fit_lane_centroids()` (lines 152-186). I first added a method to better locate the center of the peaks of the convolution using `find_peak_center()` (lines 33-55). I added sanity checks to the code that would check for pixel intensity (confidence metric) and predicted center, and then decide whether to allow shifts of the centroids at each level. The allowed shifts were scaled for each level to allow for curvature while traversing from bottom to top of image. I added a 'bootstrap' option to be used when recovery from an unknown state is required. I split the function up so it would run on one line (left or right) at a time. The function `fit_lane_centroids()` takes the centroids and confidence metrics from `find_lane_centroids()` and returns the final pixels along with polynomial fits and coefficients. Note that in the final version of the code I compute the polynomial coefficients and fits in the class LaneLine() from the raw pixels returned from `fit_lane_centroids()`. The function `fit_lane_centroids()` has a threshold option to only include centroids with high confidence (high pixel count) in the polynomial fit.

An example output showing the centroid locations, pixels included, and polynomial fit to the data is shown below:

![alt text][figure5]

****5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.****

I did the curvature in lines 332 through 351 in my code in `video_pipeline.py` inside of the `LaneLine()` class. The function is `get_curvature()`. I modified the class lecture code to do this. I found an error in the class lecture code for the values for xm_per_pix and for ym_per_pix. They were both wrong by a factor of two in opposite directions. The correct values are xm_per_pix=3.7/350 meters per pixel and ym_per_pix = 3.0/144 meters per pixel. I verified these by checking both horizontal lane width and vertical lane feature regulations for US highways. I posted on waffle issue 601.

The offset was calculated in lines 120-124 in `video_pipeline.py` inside of the `LaneFrame()` class. Offset was calculated using the fitted  x values from each line and measuring the difference at the bottom of the image.

****6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.****

I implemented this step in lines 407 through 415 in my code in `video_pipeline.py` in the functions `fill_lane_poly()` in `sliding_window.py` and in `put_text()` in `video_pipeline.py` in the class `LaneFrame()`.   Here is an example of my result on a frame from the project video:

![alt text][figure6]

---

***Pipeline (video)***

****1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).****

Here's a [link to my video output](./test_out.mp4)


***Discussion***

****1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?****

The project had several difficult problems to solve: revealing the lanes, capturing  correct pixels for line fitting, and developing a proper sequence algorithm for rejecting bad frames and keeping good ones.

****Revealing Lines****

No combinations of masks worked ideally on all images so this was a process of trial and error and rather time consuming. I settled on the simplest approach that worked for the project video. I would need to spend more time on this area to make the code work for the challenge video. The challenge video has road repair that shows up as bright image in my current binary mask. I experimented with removing dark pixels and this helped but I ran out of time to pursue the challenge video further.

****Capturing correct pixels****

The current algorithm has a lot of hard coded values that work but would ideally be adaptively trained and set by road conditions. That would be for future work. The algorithm can lock to the wrong pixels if there is either a bright patch of noise or if the visibility of the line is poor. I could envision doing a search in both directions (top down and bottom up) and am sure this could improve things. The current algorithm can get off from the start and stay off. Coming from both directions and adding some heuristics to merge the results would improve reliability.

****Sequence Algorithm****

I used an algorithm to only keep detected lines (higher confidence) in the buffer used to calculate the final fit for the line. I used several metrics to determine if a line was detected including pixel intensity (confidence),whether current fit was similar to last best fit, offset, and a check on min and max distance of the two lines. I also used a separate sanity check on bottom and top portions of each line fit. I used the previous fit to find the pixels as long as a certain threshold of frames had not already failed. I had two ways to recover lost lines. One was to simply use the sliding window histogram whenever the line had failed for more than a threshold count in  a row. Then another reset check that would kick in if another higher failure count was exceeded for either line. The reset would use the 'bootstrap' option in the sliding window code.

I also added a warning to my algorithm to alert the user that the system is in reset. I paint the lane red and print a warning on the image if the system is in reset. Meanwhile lanes are still painted. You can observe this happening on the second cement patch on two frames. It would be easy to just not show this and paint it green, but I think it is important  for a real system to display and record warnings and failures. Note that in this case, no failure  occurred in that the system recovered quickly enough that the lines were still in correct locations. See example below:

![alt text][figure7]
