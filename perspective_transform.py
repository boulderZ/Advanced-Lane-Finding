import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
from reveal_lanes_utils import *

def get_src_dst():
    '''
    Function to store and returnthard coded values for source and destination
    to be used in perspective transforms.
    '''
    OFFSET = 250
    SRC = np.float32([
        (230, 703),  # 230,703
        (573, 466),  # 574,466
        (709, 466),  # 708,466
        (1080, 703)])  # 1082,703
    DST = np.float32([
        (SRC[0][0] + OFFSET, 720),
        (SRC[0][0] + OFFSET, 0),
        (SRC[-1][0] - OFFSET, 0),
        (SRC[-1][0] - OFFSET, 720)])
    return SRC,DST

def init_perspective_transforms():
    '''
    Function returns both the forward and inverse m values needed for
    cv2.warpPerspective()
    '''
    SRC,DST = get_src_dst()
    M_FORWARD = cv2.getPerspectiveTransform(SRC, DST)
    M_INVERSE = cv2.getPerspectiveTransform(DST, SRC)
    return M_FORWARD, M_INVERSE

##################### Utility functions for debug ##################
def warp_perspective(img,inverse=False):
    if inverse:
        return cv2.warpPerspective(img, init_perspective_transforms()[1], (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    else:
        return cv2.warpPerspective(img, init_perspective_transforms()[0], (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

def process_single_file_image(filename):
    dist_pickle = pickle.load( open('camera_cal/wide_dist_pickle.p','rb'))
    img = mpimg.imread(filename)
    undst = cv2.undistort(img, dist_pickle['mtx'],dist_pickle['dist'], None, dist_pickle['mtx'])
    mask = reveal_lanes(undst)
    w_mask = warp_perspective(mask)
    return w_mask

def process_single_image(img):
    dist_pickle = pickle.load( open('camera_cal/wide_dist_pickle.p','rb'))
    undst = cv2.undistort(img, dist_pickle['mtx'],dist_pickle['dist'], None, dist_pickle['mtx'])
    mask = reveal_lanes(undst)
    w_mask = warp_perspective(mask)
    return w_mask

def undistort_single_image(img):
    dist_pickle = pickle.load( open('camera_cal/wide_dist_pickle.p','rb'))
    return cv2.undistort(img, dist_pickle['mtx'],dist_pickle['dist'], None, dist_pickle['mtx'])

if __name__ == "__main__":
    '''
    Testing only. Plots test images before and after transforms.
    '''

    SRC,DST = get_src_dst()

    l_bot = (int(DST[0][0]),int(DST[0][1]))
    l_top = (int(DST[1][0]),int(DST[1][1]))
    r_bot = (int(DST[3][0]),int(DST[3][1]))
    r_top = (int(DST[2][0]),int(DST[2][1]))

    src_l_bot = (int(SRC[0][0]),int(SRC[0][1]))
    src_l_top = (int(SRC[1][0]),int(SRC[1][1]))
    src_r_bot = (int(SRC[3][0]),int(SRC[3][1]))
    src_r_top = (int(SRC[2][0]),int(SRC[2][1]))

    #M_FORWARD,M_INVERSE = init_perspective_transforms()  # global variables
    dist_pickle = pickle.load( open('camera_cal/wide_dist_pickle.p','rb'))
    #img = mpimg.imread('test_images/straight_lines2.jpg')

    filenames = glob.glob('test_images/test*.jpg')
    print(filenames)

    img_array = []
    titles = []
    for i in range(len(filenames)):
        #print(filenames[i])
        img = mpimg.imread(filenames[i])

        undst = cv2.undistort(img, dist_pickle['mtx'],dist_pickle['dist'], None, dist_pickle['mtx'])
        img_array.append(undst)
        titles.append('undst')

        mask = reveal_lanes(undst)
        img_array.append(mask)
        titles.append('mask')


        w_img = warp_perspective(undst)
        img_array.append(w_img)
        titles.append('warped image')

        w_mask = warp_perspective(mask)
        img_array.append(w_mask)
        titles.append('warped mask')

    plot_subplot_images(img_array,titles=titles,cmap='gray',rows = 6,transpose=1 )
    '''
    #plt.imshow(w_img)
    plt.imshow(cv2.line(w_img,l_bot,l_top,color=[255, 0, 0],thickness = 3))
    plt.imshow(cv2.line(w_img,r_bot,r_top,color=[255, 0, 0],thickness = 3))
    plt.figure()
    plt.imshow(w_mask,cmap='gray')
    plt.show()
    '''
