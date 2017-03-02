import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
input: image, HOG orientations, pixels per cell, cells per block (for normalization)
visualize def. false, feature vec (all along single access True)

output: the features (and potentially the hog image itself)

TODO: 
1. change this to only do bottom half
2. set transform_sqrt=True and remove neg numb
'''
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        #TODO: set transform_sqrt=True and remove neg numb
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        #TODO: set transform_sqrt=True and remove neg numb
        return features

'''
Convert image to one-d feature fector (with resize based on bins)--getting super small sample of img

TODO: test if we can get the spatial resolution lower with same accuracy-have tried 20/20
'''
def bin_spatial(img, size=(32, 32)):
    small_img = cv2.resize(img, size)
    features = small_img.ravel()
    return features

'''
Define a function to compute color histogram features 
NEED TO CHANGE bins_range if reading .png files with mpimg!
'''
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # print('histfeatures shape', hist_features.shape)
    # hist_features = np.reshape(hist_features, (1, hist_features.shape[0]))
    # print('hist features after reshape', hist_features.shape)
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

'''
Define a function to extract features from a list of images
Have this function call bin_spatial() and color_hist()
'''
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = cv2.imread(file)
        # plt.imshow(image)
        # plt.show()
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else: feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      

        # plt.imshow(feature_image)
        # plt.title('corrected')
        # plt.show()

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    
#TOOD: use my code with switched x and r col/row
# Define a function that takes an image, start and stop positions in both x and y, 
# window size (x and y dimensions), and overlap fraction (for both x and y)
# def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
#                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
#     # If x and/or y start/stop positions not defined, set to image size
#     if x_start_stop[0] == None:
#         x_start_stop[0] = 0
#     if x_start_stop[1] == None:
#         x_start_stop[1] = img.shape[1]
#     if y_start_stop[0] == None:
#         y_start_stop[0] = 0
#     if y_start_stop[1] == None:
#         y_start_stop[1] = img.shape[0]

#     # Compute the span of the region to be searched    
#     x_span = x_start_stop[1] - x_start_stop[0]
#     y_span = y_start_stop[1] - y_start_stop[0]
    

#     # Compute the number of pixels per step in x/y
#     x_step = xy_overlap[0] * xy_window[0]
#     y_step = xy_overlap[1] * xy_window[1]

#     # Compute the number of windows in x/y
#     x_windows = int(((x_span - xy_window[0]) // x_step) + 1)
#     y_windows = int(((y_span - xy_window[1]) // y_step) + 1)

#     total_windows = x_windows * y_windows
    
#     # Initialize a list to append window positions to
#     window_list = []
#     # Loop through finding x and y window positions
#     x_start = xy_window[0]
#     y_start = xy_window[1]
#     for x_window in range(0, x_windows):
#         x_pos = int(x_start + x_window * x_step)
#         # print('first', x_pos - x_start)
#         x_first = x_pos - x_start
#         for y_window in range(0, y_windows):
#         # Calculate each window position
#             y_pos = int(y_start + y_window * y_step)
#             y_first = y_pos - y_start
#             # print('xpos ypos', ((x_first, y_first), (x_pos, y_pos)))
#             window_list.append(((x_first, y_first), (x_pos, y_pos)))

#     return window_list

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-nx_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list



# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy




'''
get all of the hog features from lower half of the image
TODO: apply to lower half
'''
# orient = 0
# pix_per_cell = 8
# cell_per_block = 2


# feature_array = hog(img, orientations=orient, \
#   pixels_per_cell=(pix_per_cell, pix_per_cell),\
#   cells_per_block=(cell_per_block, cell_per_block),\
#   visualize=False, feature_vector=False)
'''
output feature_array has shape (n_yblocks, n_xblocks, 2, 2, 9)
where n_yblocks and n_xblocks determined by shape of 
region o finterest (how many blcoks fit across and down image)


when extract from 64x64 at 2 per block,
want to extract subarrays of shape (7, 7, 2, 9)
and use np.ravel to unroll the feature vector
'''


'''
TODO: use sliding windows to find which hog features are in
which window
'''