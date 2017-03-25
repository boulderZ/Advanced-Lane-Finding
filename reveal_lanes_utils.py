import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#image = mpimg.imread('signs_vehicles_xygrad.png')

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=None,colormap=None):
    '''Implements Sobel threshold using cv2.Sobel(). Modified from lecture notes
     to include colormap options to use only l or s channel
    '''
    if ( (len(img.shape) > 2 ) and (colormap == None) ) or ((len(img.shape)==2) and colormap):
         print('ERROR: abs_sobel_thresh(): colormap  and  len(img) mismatch ')
         return None
    if colormap == None:
        color_img = img  # function called with colormap already implemented
    elif colormap == 'gray': # use grayscale
        color_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # use RGB
    elif colormap == 'hLs': # use l channel
        hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
        color_img = hls[:,:,1]  # l channel
    elif colormap == 'hlS':  # use s channel
        hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
        color_img = hls[:,:,2]  #  s channel
    else:
        print('ERROR: abs_sobel_thresh() invalid colormap given')
        return None
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(color_img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(color_img, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    out = scaled_sobel
    if thresh: # if thresh given return binary mask
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        out = grad_binary
    return out

def mag_thresh(img, sobel_kernel=3, thresh=None):
    '''
    Returns sobel magnitude with threshold option. Not used in final submit.
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # assumes RGB input
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    out = gradmag
    # Create a binary image of ones where threshold is met, zeros otherwise
    if thresh: # if thresh given, return binary mask
        mag_binary = np.zeros_like(gradmag)
        mag_binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
        out = mag_binary
    return out

def dir_threshold(img, sobel_kernel=3, thresh=None):
    '''
    Calculates directional sobel with threshold. Not used in final submit
    '''
    # Calculate gradient direction
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # calculate gradient direction
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    out = absgraddir
    if thresh:
        dir_binary =  np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        out = dir_binary
    return out

def plot_subplot_images(img_array,titles=None,cmap=None,transpose=0,
                    fontsize=10,figheight=12,rows = 0,wspace=.1,hspace_add=0):
    '''
    Imports:
        import numpy as np
        import matplotlib.pyplot as plt
    Description:
        Create grid of images with optional titles
        got help here:       http://stackoverflow.com/q/42475508/7447161
    Inputs:
        img_array = array of images to plot
        titles = (optional) list of titles
        rows = number of rows to plot, default assumes square matrix
        transpose = flip rows for columns,
        hspace_add = add more space for printing titles if wspace=0
    '''
    aspect = img_array[0].shape[0]/float(img_array[0].shape[1])
    #print( aspect)
    num_images = len(img_array)
    if transpose and rows:  # flip rows for columns
        cols = rows
        rows = int(num_images/cols)
        n = rows
        m = cols
        new_array=[]
        new_titles=[]
        for i in range(rows):
            for j in range(cols):
                new_array.append(img_array[i+j*rows])
                new_titles.append(titles[i+j*rows])
        img_array = new_array
        titles = new_titles
    elif rows:
        n = rows
        m = int(num_images/rows)
        if num_images % rows:
            print('len(img_array) = ',num_images,' not divisible by rows, truncating ...')
    else:
        grid = int(np.sqrt(num_images))  # will only show all images if square
        n = grid
        m = grid
        if num_images % grid:
            print('img_array not square, truncating ...')

    bottom = 0.1; left=0.05
    top=1.-bottom; right = 1.-left
    fisasp = (1-bottom-(1-top))/float( 1-left-(1-right) )
    #widthspace, relative to subplot size
    #wspace=0.1  # set to zero for no spacing
    hspace=wspace/float(aspect) + hspace_add #
    #fix the figure height
    #figheight= 3 # inch
    figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp

    fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right,
                        wspace=wspace, hspace=hspace)
    im = 0
    for ax in axes.flatten():
        ax.imshow(img_array[im],cmap=cmap)
        if titles:
            title = titles[im]
            ax.set_title(title,fontsize=fontsize)
        ax.axis('off')
        im += 1

    plt.show()

def extract_high_values(img, p=99.9):
    """
    Description:
        Generates a mask returning only very high pixel values for
        that channel. Used for extracting white lane lines. Using red channel
        from RGB seems to work best for that. Modified from post by pkern90 on
        github.
    Imports:
        import numpy as np
        import cv2
    Inputs:
        img = single channel image with pixels in range 0-255
        p = percentile of pixels
    Outputs:
        mask = mask with only pixels in high value range, zero else
    """
    p = int(np.percentile(img, p) - 30)
    mask = cv2.inRange(img, p, 255)
    return mask

def reveal_lanes(img, s_thresh=(170, 255), l_sx_thresh=(15, 255),remove_dark=0):
    '''
    Description:
        Function to enhance the visibility of lanes for further processing
    Imports:
        import cv2
        import numpy as np
    Inputs:
        img = RGB image to be processed
    Output:
        mask = single channel binary mask with enhanced lanes
    '''
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float) # mpimg.imread
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x of l channel and threshold
    l_scaled_sx = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=3, thresh=None)
    l_sx_binary = np.zeros_like(l_scaled_sx)
    l_sx_binary[(l_scaled_sx >= l_sx_thresh[0]) & (l_scaled_sx <= l_sx_thresh[1])] = 1

    highlights = extract_high_values(img[:,:,0])
    high_binary = np.zeros_like(highlights)
    high_binary[highlights == 255] = 1

    yellow_binary = extract_yellow(img)  # get binary mask for yellow

    # # Threshold color channel (not used in end)
    # s_binary = np.zeros_like(s_channel)
    # s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # # try removing dark  helped some on challenge but also made worse
    if remove_dark:
        dark_mask = extract_dark(img)
        dark_binary = np.zeros_like(dark_mask)
        dark_binary[dark_mask==0] = 1

    combined = np.zeros_like(l_sx_binary)
    #combined[ (l_sx_binary == 1)  | (high_binary == 1) | (yellow_binary == 255) | (s_binary==1)] = 1
    combined[ (l_sx_binary == 1)  | (high_binary == 1) | (yellow_binary == 255) ] = 1
    #combined[ (l_sx_binary == 1)  | (high_binary == 1)  ] = 1
    if remove_dark:
        combined = combined & dark_binary

    return combined


def reveal_lines_test(img, s_thresh=(170, 255), l_sx_thresh=(15, 255)):
    '''
    Description:
        Function to test various techniques to highlight the lane lines.
    Imports:
        import cv2
        import numpy as np
    Inputs:
        img = RGB image to be processed
    Outputs:
        binary_array = array of processed images for plotting
        titles = array of titles matching images processed.
    '''
    #img = np.copy(img)

    binary_array = []
    titles = []
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) (not used in end)

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float) # mpimg.imread
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x of l channel and threshold
    l_scaled_sx = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=3, thresh=None)
    l_sx_binary = np.zeros_like(l_scaled_sx)
    l_sx_binary[(l_scaled_sx >= l_sx_thresh[0]) & (l_scaled_sx <= l_sx_thresh[1])] = 1
    binary_array.append(l_sx_binary)
    titles.append('l_sx_binary')

    highlights = extract_high_values(img[:,:,0])
    high_binary = np.zeros_like(highlights)
    high_binary[highlights == 255] = 1
    binary_array.append(high_binary)
    titles.append('high_binary')

    yellow_binary = extract_yellow(img)  # get binary mask for yellow
    binary_array.append(yellow_binary)
    titles.append('yellow_binary')
    #white_binary = extract_white(img)    # get binary mask for white (removed)
    # sobel x of s channel and threshold (removed, not helping)
    # # Threshold color channel  # not Used
    # s_binary = np.zeros_like(s_channel)
    # s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # binary_array.append(s_binary)
    # titles.append('s_binary')

    combined = np.zeros_like(l_sx_binary)
    #combined[ (l_sx_binary == 1)  | (high_binary == 1) | (yellow_binary == 1) | (s_binary==1)] = 1
    combined[ (l_sx_binary == 1)  | (high_binary == 1) | (yellow_binary == 255) ] = 1
    binary_array.append(combined)

    titles.append('final combined')

    return binary_array,titles

def extract_color(img,lower,upper,colormap='rgb'):
    '''
    Imports:
        import cv2
        import numpy as np
    Description:
        Assumes img is RGB read with mpimg.imread() so RGB format
        Used to search for good color extraction for yellow/white
        Not used in final submit.
    Inputs:
        img = image in RGB
        lower = [p1,p2,p3] pixel values for lower range
        upper = [p1,p2,p3] pixel values for lower range
    '''
    #img_orig = np.copy(img)
    color_dict={'hsv':cv2.COLOR_RGB2HSV,'hls':cv2.COLOR_RGB2HLS,'rgb':None}
    # Convert RGB t
    if colormap not in color_dict:
        print('ERROR: extract_color(): colormap not in dictionary')
        return
    if color_dict[colormap]:
        color = cv2.cvtColor(img, color_dict[colormap])
    else:
        color = img
        print('RGB')
    # define range of color in pixels (a,b,c)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    # Threshold the image to get only desired color range
    mask = cv2.inRange(color, lower, upper)  # black and white
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask) # extracted color
    return mask,res

def extract_yellow(img,return_mask=1):
    '''
    Imports:
        import cv2
        import numpy as np
    Description:
        Assumes img is RGB read with mpimg.imread() so RGB format
        Finds yellow color in image and returns it and mask
    Inputs:
        img = image in RGB
    '''
    color = cv2.cvtColor(img, cv2.COLOR_RGB2HSV ) # use HSV
    # define range of color in pixels (a,b,c)
    lower = np.array([0,100,100])  # good color range for yellow in HSV
    upper = np.array([50,255,255])
    # Threshold the image to get only desired color range binary
    mask = cv2.inRange(color, lower, upper)  # black and white
    # Bitwise-AND mask and original image
    if return_mask:
        out = mask
    else:
        out = cv2.bitwise_and(img,img, mask= mask) # extracted color
    return out

def extract_white(img,return_mask=1):
    '''
    Imports:
        import cv2
        import numpy as np
    Description:
        Assumes img is RGB read with mpimg.imread() so RGB format
        Finds yellow color in image and returns it and mask.
        Not used in final submit
    Inputs:
        img = image in RGB
    '''
    color = cv2.cvtColor(img, cv2.COLOR_RGB2HSV ) # use HSV
    # define range of color in pixels (a,b,c) [20,0,180] to [255,80,255] orig
    lower = np.array([20,0,180])  # good color range for white in HSV
    upper = np.array([255,80,255])
    # Threshold the image to get only desired color range binary
    mask = cv2.inRange(color, lower, upper)  # black and white
    # Bitwise-AND mask and original image
    if return_mask:
        out = mask
    else:
        out = cv2.bitwise_and(img,img, mask= mask) # extracted color
    return out

def extract_dark(img):
    '''
    Returns mask of dark pixels. Use below to get non-dark
    # dark_mask = extract_dark(img)
    # dark_binary = np.zeros_like(dark_mask)
    # dark_binary[dark_mask==0] = 1
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0.), (255, 153, 128))
    return mask

def plot_pipeline():
    '''
    Used for testing.
    '''
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(5, 120))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(5, 120))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=(30,100))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(.7, 1.7))

    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    img_array = []
    img_array.append(gradx)
    img_array.append(mag_binary)
    img_array.append(dir_binary)
    #plt.imshow(gradx,cmap='gray')
    combined[( (gradx == 1) ) & ((mag_binary == 1) & (dir_binary == 1))] = 1
    img_array.append(combined)
    #plt.imshow(combined,cmap='gray')
    #plt.show()
    titles = ['gradient x','magnitude','direction','combined']
    plot_subplot_images(img_array,titles=titles,cmap='gray',
                        fontsize=10,figheight=8,rows = 0,wspace=.1,hspace_add=0)
    return
