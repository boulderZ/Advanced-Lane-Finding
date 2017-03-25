import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
from moviepy.editor import VideoFileClip
from reveal_lanes_utils import *
from perspective_transform import *

def get_curvature_poly(funcx,ploty):
    '''
    Returns calculated curvature given original poly function and y values.
    Performs conversion from pixels to meters
    Only used in testing individual blocks in sliding_window.py
    funcx is np.poly1d(orig_fit)
    ploty is the desired y values to use
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3/144 # meters per pixel in y dimension 30/720 orig wrong
    xm_per_pix = 3.7/350 # meters per pixel in x dimension, 3.7/700 orig
    y_eval = 719
    y_pixels = ploty
    x_pixels = funcx(y_pixels)
    fit_cr = np.polyfit(y_pixels*ym_per_pix, x_pixels*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad

def window_mask(width, height, img_ref, center,level):
    '''
    Returns window centered on center for masking pixels during lane line
    finding algorithm. Called by fit_lane_centroids().
    '''
    output = np.zeros_like(img_ref)
    height_start = int(img_ref.shape[0]-(level+1)*height)
    height_stop = int(img_ref.shape[0]-level*height)
    width_start = max(0,int(center-width/2))
    width_stop = min(int(center+width/2),img_ref.shape[1])
    output[height_start:height_stop,width_start:width_stop] = 1
    return output

def find_peak_center(conv,init_center,margin,offset):
    '''
    Description:
        Basically does a full width half max type of operation. Assumes a fair
        amount about the input conv. Should have smooth sides at 0.75 of the
        max peak value. Finds the indices to either side of the peak at
        approximate values of 0.75 of max peak value in conv. Then finds and
        returns the center estimated from this width measurement.
    Inputs:
        conv = waveform with peak, result of convolution typically
        init_center = initial guess of center produced by argmax
        offset = lenght of window/2
        margin = +/- around peak to search
    Outputs:
        new_center = computed center value
    '''
    if (init_center-margin+offset < 0) or (init_center+margin+offset > len(conv)):
        return 0
    fwhm = 0.75*np.max(conv[init_center-margin+offset:init_center+margin+offset])
    idx=(np.abs(conv[init_center-margin+offset:init_center+offset]-fwhm)).argmin() + init_center-margin+offset
    idx2=(np.abs(conv[init_center+offset:init_center+margin+offset]-fwhm)).argmin() + init_center+offset
    new_center = int((idx2-idx)/2) + idx - offset
    return new_center

def find_lane_centroids(warped,lane = None,bootstrap=0):
    '''
    Description:
        Use convolution windows to find centers of lane sections. Modified from
        class notes substantially. Combined with a peak center locator
        (find_peak_center()) improves center estimation. Also, confidence
        metrics added to determine to keep  new center or old center and also to
        limit possible shift of line segment center. Currently uses fixed
        settings but could be made adaptive.
    Inputs:
        warped = image with lane lines enhanced and birds-eye view
        lane = 'right' or 'left' determines which side
        bootstrap = flag to use  fixed center locations at  start for recovery
    Outputs:
        window_centroids = center location of pixels found at each level
        window_confidence = max value of convolution over found center
    '''
    if lane == 'left':
        start = 370
        end = 640
        guess = 504
        min_center = 420 # was (420,600) mod allow slower change
        max_center = 600 # was (600,800)
    elif lane == 'right':
        start = 640
        end = 1050
        guess = 852
        min_center = 740
        max_center = 1000
    else:
        assert lane in ['left','right'], 'find_window_centroids() need side'
    min_thresh = 700.0  # used for sanity checks
    max_window_shift = 40  # if shift rqquested too large, ignore it
    window_width = 50  # width of convolution filter
    window_height = 80 # height of convolution filter
    offset = int(window_width/2)  # offset convolution result
    margin = 100   #  +/= horizontal search value from estimated center
    img_width = warped.shape[1]
    img_height = warped.shape[0]
    img_half_width = int(img_width/2)
    window_centroids = [] # Store the (left,right) window centroid positions
    window_confidence = [] # store confidence metrics
    window = np.ones(window_width) # Create window template
    # Sum quarter bottom of image to get slice. Then convolve with window
    # and find the center of the peak

    sum = np.sum(warped[int(2*img_height/4):,start:end], axis=0)
    conv = np.convolve(window,sum)
    center = np.argmax(conv) - offset
    new_center = find_peak_center(conv,center,margin,offset)
    if new_center and (min_center < new_center+start < max_center):
        max_conv = np.max(conv[new_center-margin+offset:new_center+margin+offset])
        center = new_center + start
    else:
        center = guess
        new_center = guess - start
        max_conv = np.max(conv[new_center-margin+offset:new_center+margin+offset])

    if bootstrap:  # use when algorithm is failing/lost
        sum = np.sum(warped[int(2*img_height/4):,start:end], axis=0)
        conv = np.convolve(window,sum)
        center = guess
        new_center = guess - start
        max_conv = np.max(conv[new_center-margin+offset:new_center+margin+offset])
    # Add what we found for the first layer
    window_centroids.append(center)
    window_confidence.append(max_conv)

    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(img_height/window_height)):
        margin = 80 # try reducing margin from 100
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
                     warped[int(img_height-(level+1)*window_height):
                     int(img_height-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        min_index = int(max(center+offset-margin,0))
        max_index = int(min(center+offset+margin,img_width))
        old_center = center  # save for later
        if np.max(conv_signal[min_index:max_index]) < min_thresh:
            center = center  # no confidence, keep last one
            max_conv = np.max(conv_signal[min_index:max_index])
        else:
            center = np.argmax(conv_signal[min_index:max_index])+min_index-offset
            # restrict maximum shift only if confidence high from previous
            if (abs(center-old_center) > max_window_shift) and max_conv > 1000.0:
                center = old_center
            new_center = find_peak_center(conv_signal,center,margin,offset)
            if new_center and (min_center-(level*25) < new_center < max_center+(level*25)):
                max_conv = np.max(conv_signal[new_center-margin+offset:new_center+margin+offset])
                center = new_center
            else:
                center = old_center
                max_conv = np.max(conv_signal[old_center-margin+offset:old_center+margin+offset])
        # Add what we found for that layer
        window_centroids.append(center)
        #  Add maximum value as a confidence metric to be used later
        window_confidence.append(max_conv)

    return window_centroids,window_confidence

def fit_lane_centroids(warped,window_centroids,window_confidence,keep_threshold=400.0):
    '''
    Description:
        Locates pixels within the window centroids and returns the pixels along
        with confidence metrics. Only keeps pixels if threshold for that level
        is met. Currently hard coded threshold, but could be made adaptive.
    Inputs:
        warped = binary image with lane lines enhanced
        window_centroids = center values from find_lane_centroids()
        window_confidence = confidence values from find_lane_centroids()
        keep_thresh = reject level if confidence below  this threshold
    Outputs:
        out_vars = (fitx,ploty,fit,x_array,y_array,conf_data)
        fitx = polynomial fit x values from fit coefficients
        ploty = y values used to get fitx
        fit = polynomial coefficients
        x_array = x pixel locations
        y_array = y pixel locations
    '''
    #keep_threshold = 400.0 # reject pixels if less, 400 (adaptive TBD)
    x_array = []
    y_array= []
    conf_index = []
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
        mask = window_mask(50,80,warped,window_centroids[level],level)
        lane_mask = (mask == 1) & (warped)  # overlap of mask and image
        nonzerox = np.array(lane_mask.nonzero()[1])
        nonzeroy = np.array(lane_mask.nonzero()[0])
        # reject pixels if confidence is low (make adaptive TBD)
        if window_confidence[level] >= keep_threshold:
            x_array.append(nonzerox)
            y_array.append(nonzeroy)
            conf_index.append(level)  # keep indices of confidence levels used

    x_array = np.concatenate(x_array)
    y_array = np.concatenate(y_array)

    #radius = get_curvature(x_array,y_array)
    conf_len = len(conf_index)
    conf_sum = np.sum(np.array(window_confidence)[conf_index])
    conf_data = (conf_len,conf_sum)
    if len(x_array):
        fit = np.polyfit(y_array, x_array, 2)
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    else:
        fit = 0
        ploty = 0
        fitx = 0
    out_vars = (fitx,ploty,fit,x_array,y_array,conf_data)
    return out_vars

########################### code for overlay lane on original
def fill_lane_poly(img_shape,left_fitx,right_fitx,ploty,in_reset=False):
    '''
    Description:
        Function to fill space between the polynomial line fits with a color
        to paint on the lane. Modified from class lecture notes.
    Inputs:
        img_shape = image.shape
        left_fitx = x values from line fit to left line
        right_fitx = x values from line fit to right line
        ploty = y values to plot x from
        in_reset = warning flag to paint red instead of green if in reset
    Outputs:
        color_warp = overlay image with lane painted in warped space
    '''
    # Create an image to draw the lines on
    warp_zero = np.zeros(img_shape).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    # draw in red if system is in reset, else green
    if in_reset:
        cv2.fillPoly(color_warp, np.int_([pts]), (255,0, 0))  # red
    else:
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))  # green
    return color_warp

def search_poly_lane(binary_warped,search_fit):
    '''
    Description:
        Find pixels only within margin of given polynomial. Modified from
        lecture notes.
    Inputs:
        binary_warped = warped  binary image with lane lines visible
        search_fit = polynomial coefficients to be used for the search_fit
    Outputs:
        poly_vars = (poly_fitx,ploty,poly_fit,x,y)
        poly_fitx = x values from the new fit
        ploty = y values used for the new fit
        poly_fit = polynomial coefficients from new fit
        x = x values for pixels found
        y = y values for pixels found
    '''
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 30  # number of pixels to either side of seach_fit poly to use
    # find nonzero values within +/- margin of the search_fit polynomial
    lane_inds = ((nonzerox > (search_fit[0]*(nonzeroy**2) +
                search_fit[1]*nonzeroy + search_fit[2] - margin)) &
                (nonzerox < (search_fit[0]*(nonzeroy**2) +
                search_fit[1]*nonzeroy + search_fit[2] + margin)))
    # extract  line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]
    # Fit a second order polynomial
    poly_fit = np.polyfit(y, x, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    poly_fitx = poly_fit[0]*ploty**2 + poly_fit[1]*ploty + poly_fit[2]
    poly_vars = (poly_fitx,ploty,poly_fit,x,y)
    return poly_vars

#####  PLOTTING ONLY #####################
def plot_window_centroids(warped,window_centroids):
    '''
    Plot routine for visualizing window centroids
    '''
    # If we found any window centers

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
        l_mask = window_mask(50,80,warped,window_centroids[level][0],level)
        r_mask = window_mask(50,80,warped,window_centroids[level][1],level)
        # Add graphic points from window mask here to total pixels found
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    # add both left and right window pixels together
    template = np.array(r_points+l_points,np.uint8)
    zero_channel = np.zeros_like(template) # create a zero color channel
    # make window pixels green
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),
                        np.uint8)
    warpage=np.dstack((warped, warped, warped))*255 # create 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay
    return output

def plot_search_poly_lines(binary_warped,left_fit,right_fit):
    '''
    Plot routine for visualizing the poly search margin
    '''
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 30
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.figure()
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
#####################################################

if __name__ == "__main__":
    ''' Below is for testing. Will display the window centroids used and the
    polynomial search margin for the same fit for any single image from the
    clip or a file. And then paint the result on the final image and display
    '''
    # Read in a thresholded image
    #warped = mpimg.imread('warped_example.jpg')
    clip = VideoFileClip("project_video.mp4")
    # 22 has first cement patch 22.16 bad, 15 straight
    #clip = VideoFileClip("challenge_video.mp4")
    img = clip.get_frame(10)
    warped = process_single_image(img)
    plt.figure()
    plt.imshow(warped,cmap='gray')
    plt.show()

    left_centroids,left_confidence = find_lane_centroids(warped,lane='left',bootstrap=0)
    leftvars = fit_lane_centroids(warped,left_centroids,left_confidence)
    right_centroids,right_confidence = find_lane_centroids(warped,lane='right',bootstrap=0)
    rightvars = fit_lane_centroids(warped,right_centroids,right_confidence)

    left_x = leftvars[3]; left_y = leftvars[4]
    right_x = rightvars[3]; right_y = rightvars[4]

    if len(left_x) and len(right_x): # if pixels exist, proceed else print FAIL
        left_poly = np.polyfit(left_y,left_x,2)
        left_current_funcx = np.poly1d(left_poly)
        right_poly = np.polyfit(right_y,right_x,2)
        right_current_funcx = np.poly1d(right_poly)
        print('left rad new = ',get_curvature_poly(left_current_funcx,leftvars[1]))
        print('right rad new = ',get_curvature_poly(right_current_funcx,leftvars[1]))
        print(right_poly)

        color_warp = fill_lane_poly(warped.shape,left_current_funcx(leftvars[1]),
                                 right_current_funcx(leftvars[1]),leftvars[1])

        newwarp = warp_perspective(color_warp,inverse=True)

        # Combine the result with the original image
        undist = undistort_single_image(img)
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        plt.figure()
        plt.imshow(result)

        plot_search_poly_lines(warped,left_poly,right_poly)

        # Display the final results
        x = zip(left_centroids,right_centroids)
        window_centroids = [x for x in x]
        plt.figure(figsize=(12,12))
        plt.imshow(plot_window_centroids(warped,window_centroids))
        plt.plot(left_current_funcx(leftvars[1]), leftvars[1], color='red')
        plt.plot(right_current_funcx(leftvars[1]), leftvars[1], color='red')

        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.title('sliding window fitting results')
        plt.figure()
        plt.imshow(img)
        plt.show()
    else:
        print('FAILED to get any pixels: left, right = ',len(left_x),len(right_x))
