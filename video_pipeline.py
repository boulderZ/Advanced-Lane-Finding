import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
from moviepy.editor import VideoFileClip
from reveal_lanes_utils import *
from perspective_transform import *
from sliding_window import *

class SetParameters:
    n_frames = 5
    img_shape = (720,1280)
    img_bot = img_shape[0] - 1
    ym_per_pix = 3/144 # meters per pixel in y  dimension
    xm_per_pix = 3.7/350 # meters per pixel in x dimension
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0] )
    m_forward,m_inverse = init_perspective_transforms()
    def __init__(self,run_calibration =0):
        self.run_calibration = run_calibration
        if self.run_calibration:
            # re-run camera calibration instead of loading
            # set save_pickle =0 to avoid overwriting existing pickle file
            self.cam_mtx,self.cam_dist = calibrate_camera(save_pickle =0)
        else:
            # get camera matrix and camera distortion coefficients from pickle
            dist_pickle = pickle.load( open('camera_cal/wide_dist_pickle.p','rb'))
            self.cam_mtx = dist_pickle['mtx']
            self.cam_dist = dist_pickle['dist']

class LaneFrame:
    def __init__(self):
        self.framenumber = 0
        self.test_fail_array = []
        self.curvature = None
        self.curvature_array = []
        self.current_offset = None
        self.best_offset = None
        self.offset_array = []

        #parallel_thresh=(0.0003, 0.55), dist_thresh=(350, 460)
    def distance_accept(self,left_line,right_line):
        '''
        Checks if lane lines are correct distance apart. Also serves to check
        whether lanes are parallel. TBD: convert to meters
        '''
        diffx = abs(left_line.current_fitx-right_line.current_fitx)
        if (np.max(diffx) > 415) or (np.min(diffx) < 300):
            # if 90 < self.framenumber < 105:
            #     print('np max = ',np.max(diffx),'np min = ',np.min(diffx))
            return False
        else:
            return True
    def update_current_offset(self,left_line,right_line):
        leftval = left_line.current_fitx[param.img_bot]
        dist = (right_line.current_fitx[param.img_bot] - leftval)/2
        offset = (dist + leftval - param.img_shape[1]/2) * param.xm_per_pix
        self.current_offset = offset

    def offset_accept(self,left_line,right_line):
        '''
        Checks if absolute value of offset is less than threshold in meters.
        Average car width is approximately 1.85 meters  (US lane widths are
        3.7 meters). Using a little less than half of that for threshold.
        '''
        leftval = left_line.current_fitx[param.img_bot]
        dist = (right_line.current_fitx[param.img_bot] - leftval)/2
        offset_bot = (dist + leftval - param.img_shape[1]/2) * param.xm_per_pix
        if abs(offset_bot) < 0.75 :
            return True
        else:
            return False

    def update_bot_top_left(self,left_line):
        '''
        Sanity checks on fitx at top and bottom of image. Top range is larger to
        accomodate curvature. TBD: convert to meters
        '''
        if (440 < left_line.current_fitx[param.img_bot] < 564) and \
                (220 < left_line.current_fitx[0] < 750):
            left_line.bot_top = True
        else:
            left_line.bot_top = False

    def update_bot_top_right(self,right_line):
        if (780 < right_line.current_fitx[param.img_bot] < 920) and \
                  (540 < right_line.current_fitx[0] < 1100):
            right_line.bot_top = True
        else:
            right_line.bot_top = False

    def recover(self,line,warped,lane=None):
        '''
        Description:
            Attempts to recover line detection from scratch when a reset is
            requested. Either uses looser criteria for existing fit or calls
            the histogram  routine with bootstrap option.
        '''
        if line.bot_top and line.confidence > 3000:
            line.recent_fitx = [line.current_fitx]
            line.recent_poly = [line.current_poly]
            #print('in recover if: framenumber = ',self.framenumber)
        else:
            # run histogram with bootstrap option
            centroids,confidence = find_lane_centroids(warped,lane=lane,
                                                           bootstrap = 1)
            vars = fit_lane_centroids(warped,centroids,confidence)
            line.recent_fitx = []
            line.recent_poly = []
            line.back2back = False
            line.update_current(vars[3],vars[4])
            #print('in recover hist framenumber = ',self.framenumber)

    def update_curvature(self,left_line,right_line):
        self.curvature = (left_line.best_curvature + right_line.best_curvature)/2.
        if self.curvature > 10000.0: # case of very small 1st coefficient
            self.curvature = 10000.0

    def update_offset(self,left_line,right_line):
        leftval = left_line.best_fitx[param.img_bot]
        dist = (right_line.best_fitx[param.img_bot] - leftval)/2
        offset = (dist + leftval - param.img_shape[1]/2) * param.xm_per_pix
        self.best_offset = offset

    def check_lines(self,left_line,right_line,warped):
        '''
        Description:
            Checks if lines should be marked as detected. If detected, include
            in best buffer for averaging, if not reject. Keeps track of number
            of rejections and will cause a reset if too many frames are missed
            in a row. TBD: replace hard coded values with adaptive/computed
            values on the fly.
        '''
        # first check if reset required
        if right_line.line_reject_count > 6:
            right_line.reset = 1
        if left_line.line_reject_count > 6:
            left_line.reset = 1
        if left_line.reset:
            self.update_bot_top_left(left_line)
            self.recover(left_line,warped,lane='left')
        if right_line.reset:
            self.update_bot_top_right(right_line)
            self.recover(right_line,warped,lane='right')

        # run line checks only if both lines exist (confidence > 0)
        if left_line.confidence and right_line.confidence:
            self.update_bot_top_left(left_line)
            self.update_bot_top_right(right_line)

            distance_good = self.distance_accept(left_line,right_line)
            self.update_current_offset(left_line,right_line)
            self.offset_array.append(self.current_offset)

            offset_good = self.offset_accept(left_line,right_line)
            # Apply heuristics to determine if line is detected
            if distance_good and offset_good:
                combo_good = True
            else:
                combo_good = False
            # try to detect left line
            #if left_line.solid_line and offset_good:
            if left_line.solid_line and distance_good:
                # ignore combo result, highly likely good
                left_line_good = 1
            elif combo_good and left_line.back2back:
                left_line_good = 1
            elif combo_good and left_line.confidence > 4000.0: # TBD: adaptive
                left_line_good = 1
            else:
                left_line_good = 0
            # try to detect right line
            #if right_line.solid_line and offset_good:
            if right_line.solid_line and distance_good:
                # ignore combo result, highly likely good
                right_line_good = 1
            elif combo_good and right_line.back2back:
                right_line_good = 1
            elif combo_good and right_line.confidence > 4000.0:
                right_line_good = 1
            else:
                right_line_good = 0
        else:                      # if either line has zero confidence
            left_line_good = 0
            right_line_good = 0
            combo_good = False
            offset_good = False
            distance_good = False
        # update left line
        if left_line_good:
            left_line.detected = True
            left_line.line_reject_count = 0
            left_line.update_histogram = False
            left_line.reset = 0
        else:
            left_line.detected = False
            # reject last frame from best averages  only if line was added
            if len(left_line.recent_poly) > 1 and left_line.confidence:
                left_line.recent_poly = left_line.recent_poly[:-1]
                left_line.recent_fitx = left_line.recent_fitx[:-1]
            left_line.line_reject_count += 1
            left_line.bad_frame_index.append(self.framenumber) # debug

            # if missed more than 4 frames, update with histogram search
            if left_line.line_reject_count > 3:
                left_line.update_histogram = True
        # update right line
        if right_line_good:
            right_line.detected = True
            right_line.line_reject_count = 0
            right_line.update_histogram = False
            right_line.reset = 0
        else:
            right_line.detected = False
            if len(right_line.recent_poly) > 1 and right_line.confidence:
                right_line.recent_poly = right_line.recent_poly[:-1]
                right_line.recent_fitx = right_line.recent_fitx[:-1]
            right_line.line_reject_count += 1
            right_line.bad_frame_index.append(self.framenumber)

            if right_line.line_reject_count > 3:
                right_line.update_histogram = True
        # debug
        self.test_fail_array.append((left_line.detected,right_line.detected,combo_good,right_line.confidence,distance_good,offset_good))

        self.framenumber += 1

    def put_text(self,img,in_reset):
        '''
        Function to place text on image showing radius of curvature and offset
        in meters. Modified from pkern post on Github.
        '''
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Radius of Curvature = %d(m)' % self.curvature,
                    (50, 50), font, 1, (255, 255, 255), 2)
        side = 'left' if self.best_offset < 0 else 'right'
        cv2.putText(img,
             'Vehicle is %.2fm %s of center' % (np.abs(self.best_offset), side),
             (50, 100), font, 1, (255, 255, 255), 2)
        if in_reset:
            cv2.putText(img,
                 'SYSTEM RESET: Re-Acquiring lane lines... ' ,
                 (50, 150), font, 1, (255, 0, 0), 2)


class LaneLine:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # keeps track of whether solid line detected
        self.solid_line = None
        # save pixel count as measure of confidence
        self.confidence = None
        # keeps track of how many lines were skipped
        self.line_reject_count = 0
        # flag for using histogram to update pixels
        self.update_histogram = True
        # x fit values of most recent line
        self.current_fitx = None
        #average x values of the fitted line over the last n iterations
        self.best_fitx = None
        # buffer of recently kept x values of fitted line
        self.recent_fitx = []
        # polynomial coefficents of the last kept n fits
        self.recent_poly = []
        #polynomial coefficients averaged over the last n iterations
        self.best_poly = None
        #polynomial coefficients for the most recent fit
        self.current_poly = None
        #radius of curvature of the line in some units
        self.current_curvature = None
        self.best_curvature = None
        self.curvature_array = []
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # more stats
        self.bad_frame_index = []
        self.conf_array = []
        # flag for whether previous and current fits are similar
        self.back2back = False
        # flag set if lane lines have not been detected for too long
        self.reset = 0  # flag to warn bad results and attempt recovery
        # one of the acceptance criteria for a line
        self.bot_top = False
    def update_current(self,x,y):
        assert len(x) == len(y), 'LaneLine: update() len x != len y'

        self.allx = x
        self.ally = y
        #  do any extra outlier removal  on pixels here
        #  TBD

        # update line confidence
        self.confidence = len(self.allx)
        self.conf_array.append(self.confidence) # use for debug
        if self.confidence: # only update if line exists
            if self.confidence > 10000.0:  # measure of high confidence
                self.solid_line = 1

            # update poly coefficients and fitx and store most recent
            self.current_poly = np.polyfit(self.ally,self.allx,2)
            self.current_fitx = np.poly1d(self.current_poly)(param.ploty)

            self.recent_fitx.append(self.current_fitx)
            self.recent_poly.append(self.current_poly)
            # keep most recent updates in a buffer
            if len(self.recent_fitx) > param.n_frames:
                self.recent_fitx = self.recent_fitx[1:]
                self.recent_poly = self.recent_poly[1:]
            # do line sanity checks
            if len(self.recent_fitx) > 1:
                # compare current line to last best fit
                #if self.line_accept(self.current_fitx,self.recent_fitx[-2]):
                if self.line_accept(self.current_fitx,self.best_fitx):
                    self.back2back = True
                else:
                    self.back2back = False


    def line_accept(self,cur_fitx,prev_fitx,thresh = 20.0):
        '''
        Description:
            Checks difference between previous and current line
        Inputs:
            cur_fitx = x values from fitted line from current_funcx(ploty)
            prev_fitx = x values from fitted line from prev_funcx(ploty)
            thresh = threshold in pixels to qualify as good line
        '''
        if np.max(abs(cur_fitx-prev_fitx)) > thresh:
            return False
        else:
            return True
    #def get_curvature(self,fitx,y_eval=719):
    def get_curvature(self,fitx):
        '''
        Description:
            Calculates radius of curvature in meters given x values generated
            from fit to line from ploty and pixel to meter conversions in param
            see:
            http://www.intmath.com/applications-differentiation/8-radius-curvature.php
        Inputs:
            fitx = *_funcx(ploty) where *_funcx = poly1d(polyfit(x,y))
            y_eval = point to do calculation at
        '''
        # funcx is np.poly1d(orig_fit)
        # ploty is the desired y values to use
        # Define conversions in x and y from pixels space to meters
        y_eval = param.img_bot
        fit_cr = np.polyfit(param.ploty*param.ym_per_pix,
                            fitx*param.xm_per_pix, 2)
        curverad = ((1 + (2*fit_cr[0]*y_eval*param.ym_per_pix +
                          fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        return curverad


    def update_best(self):
        ''' Update coefficients and fitx with average of best recently
        detected lines and update curvature '''
        if len(self.recent_fitx):
            self.best_poly = np.mean(self.recent_poly,axis=0)
            self.best_fitx = np.mean(self.recent_fitx,axis=0)
            self.best_curvature = self.get_curvature(self.best_fitx)
            self.curvature_array.append(self.best_curvature)
        else: # empty arrays, don't  update, put error value in debug array
            self.curvature_array.append(-10.0)



def pipeline(frame):
    orig_img = np.copy(frame)
    # undistort image using camera calibration
    undist = cv2.undistort(frame, param.cam_mtx,param.cam_dist,
                           None, param.cam_mtx)
    # Enhance lane visibility
    mask = reveal_lanes(undist)
    # warp to birds-eye view
    warped = cv2.warpPerspective(mask, param.m_forward,
             (param.img_shape[1],param.img_shape[0]), flags=cv2.INTER_LINEAR)
    # get pixels from each lane
    if left_line.update_histogram: # if first time, or reset
        left_centroids,confidence = find_lane_centroids(warped,lane='left')
        left_vars = fit_lane_centroids(warped,left_centroids,confidence)
        #left_line.update_histogram = False
    else:  # use previous polynomial fit to get pixels
        left_vars = search_poly_lane(warped,left_line.best_poly)

    if right_line.update_histogram:
        right_centroids,confidence = find_lane_centroids(warped,lane='right')
        right_vars = fit_lane_centroids(warped,right_centroids,confidence)
        #right_line.update_histogram = False
    else:
        right_vars = search_poly_lane(warped,right_line.best_poly)

    left_x = left_vars[3]; left_y = left_vars[4]
    right_x = right_vars[3]; right_y = right_vars[4]

    # Get current updates
    left_line.update_current(left_x,left_y)
    right_line.update_current(right_x,right_y)

    # Do sanity checks on both lines
    # TBD is bad_frame_count incremented?
    lf.check_lines(left_line,right_line,warped)

    # update with best recently detected
    left_line.update_best()
    right_line.update_best()
    # update curvature and offset
    lf.update_curvature(left_line,right_line)
    lf.update_offset(left_line,right_line)
    # Fill space between lane lines with color
    # set warning if either line is currently in reset
    reset_flag = left_line.reset or right_line.reset
    newwarp = fill_lane_poly(warped.shape,left_line.best_fitx,
                             right_line.best_fitx,param.ploty,reset_flag)
    # inverse warp to get back to original view
    newwarp = cv2.warpPerspective(newwarp, param.m_inverse,
             (param.img_shape[1],param.img_shape[0]), flags=cv2.INTER_LINEAR)
    # combine images to get lane overlay
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # add curvature and offset and reset  warning text to image
    lf.put_text(result,reset_flag)
    return result


if __name__ == "__main__":
    # initialize common parameters and classes
    param = SetParameters(run_calibration = 0) # =0, use existing calibration
    left_line = LaneLine()
    right_line = LaneLine()
    lf = LaneFrame()
    # get video clip
    #clip = VideoFileClip("project_video.mp4").subclip(20,26) #21.2-25.2
    clip = VideoFileClip("project_video.mp4")
    #clip = VideoFileClip("challenge_video.mp4").subclip(0,8)
    # run the clip  through the pipeline
    project_clip = clip.fl_image(pipeline)
    # write output to video file
    project_clip.write_videofile('test_out.mp4',audio=False)
    # # Debug
    # plt.plot(right_line.bad_frame_index)
    # plt.figure()
    # x = range(len(lf.test_fail_array))
    # #left line detected
    # plt.plot(x,[i[0] for i in lf.test_fail_array])
    # plt.figure()
    # #right line detected
    # plt.plot(x,[i[1] for i in lf.test_fail_array])
    # plt.figure()
    # # offset
    # plt.plot(x,[i[5] for i in lf.test_fail_array])
    # plt.figure()
    # # distance good
    # plt.plot(x,[i[4] for i in lf.test_fail_array])
    # plt.show()
