import cv2
import glob
import time
import queue
import pickle
import numpy as np
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from helpers import *

    
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    # img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb);
    # ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    window_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            # print('spatial', spatial_features.shape)
            # print('hist', hist_features.shape)
            # print('hog', hog_features.shape)
            # Scale features and make a prediction
            # test1 = np.hstack((spatial_features, hog_features)).reshape(1, -1)
            # print('test1', test1.shape)
            test_features = X_scaler.transform(np.hstack((hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bbox = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
                # print('bbox', bbox[0], bbox[1])
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                window_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

    return draw_img, window_list

'''
Combine duplicate bounding boxes and remove false positives.
1. add a pixel val for each bounding box found using a heatmap
2. apply a threshold on number of pixel vals to remove false positives (if only a few bounding boxes)
3. label the bounding boxes using scipy to identify separate boxes
4. apply new bounding boxes to each of labeled boxes
'''

'''
given an image, add a pixel val for each place place in the bounding box
'''
def increment_heatmap(heatmap, bbox_list):
    for bbox in bbox_list:
        #each bbox is of form (x1, y1), (x2, y2)
        #increment the pixel val by 1
        heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
    return heatmap

'''
apply threshold of number of pixels needed in heatmap to keep those vals
'''
def apply_thresh(heatmap, thresh):
    heatmap[heatmap <= thresh] = 0
    return heatmap

'''
apply new bounding boxes based on the labeled boxes
'''
def box_labels(img, labels):
    # for number of labels
    for label in range(1, labels[1] + 1):
        # get the vals that apply just to that label
        curr_points = (labels[0] == label).nonzero()
        # print('length', len(curr_points))
        nonzero_y = np.array(curr_points[0])
        # print('x', nonzero_x.shape)
        nonzero_x = np.array(curr_points[1])
        # print('y', nonzero_y.shape)
        # draw a box on the orig image with min and max vals
        # of form: x1, y1, x2, y2
        # if (nonzero_x.shape[0] > 0):
        bbox = ((np.min(nonzero_x), np.min(nonzero_y)), (np.max(nonzero_x), np.max(nonzero_y)))
        # print('bbox', bbox[0], bbox[1])
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    return img

class Boxes:

    def __init__(self):
        self.max_3_q = queue.Queue()

    def save_box(self, box_list):
        self.max_3_q.put(box_list)
        print('save box called, size now: ', self.max_3_q.qsize())
        if self.max_3_q.qsize() > 3:
            throw_away = self.max_3_q.get()
            print('get called to remove: ', throw_away)

    def get_three(self):
        return self.max_3_q

if __name__ == '__main__':
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = False # Spatial features on or off
    hist_feat = False # Histogram features on or off
    hog_feat = True # HOG features on or off

    # number of model
    num = '10'
    mini = False
    # boolean for training svm model or just doing sliding windows functionality
    needs_training = False
    if needs_training:

    
        notcars = glob.glob('./data/non-vehicles/Extras/*.png') + glob.glob('./data/non-vehicles/GTI/*.png')
        # print('length not cars', len(notcars))
        cars = glob.glob('./data/vehicles/GTI_Far/*.png') + glob.glob('./data/vehicles/GTI_Left/*.png') \
        + glob.glob('./data/vehicles/GTI_MiddleClose/*.png') + glob.glob('./data/vehicles/GTI_Right/*.png') \
        + glob.glob('./data/vehicles/KITTI_extracted/*.png')
        # print('length cars', len(cars))

        car_ind = np.random.randint(0, len(cars))
        car_image = mpimg.imread(cars[car_ind])
        notcar_image = mpimg.imread(notcars[car_ind])
        print('we have', len(cars), 'cars and', len(notcars),\
            ' non-cars')
        print('ofsize: ', car_image.shape, 'and data type', car_image.dtype)

        # if want to visualize car/not car, uncomment
        # fig = plt.figure()
        # plt.subplot(121)
        # plt.imshow(car_image)
        # plt.title('example car')
        # plt.subplot(122)
        # plt.imshow(notcar_image)
        # plt.title('example not-car')
        # plt.show()

        if mini:
            sample_size = 500
            cars = cars[0:sample_size]
            notcars = notcars[0:sample_size]

        car_features = extract_features(cars, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features = extract_features(notcars, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
    
        # Define the labels vector (stack horizontally)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        # rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=88)
        # random_state = rand_state

        print('Using:',orient,'orientations',pix_per_cell,
            'pixels per cell and', cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC 
        svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        # t = time.time()

        
        # save image
        output_svc = open('models/trained_model' + num + '.pkl', 'wb')
        pickle.dump(svc, output_svc)

        # save scaler
        output_scaler = open('models/x_scaler' + num + '.pkl', 'wb')
        pickle.dump(X_scaler, output_scaler)

    else: 
        load_svc = open('models/trained_model' + num + '.pkl', 'rb')
        svc = pickle.load(load_svc)

        load_scaler = open('models/x_scaler' + num + '.pkl', 'rb')
        X_scaler = pickle.load(load_scaler)
        # run below to check accuracy, but have to change code to load X_test images
        # print('after save, Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # taken from one of images originally

    # image = mpimg.imread('data/vehicles/GTI_Far/image0000.png')
    image = mpimg.imread('test_images/test1.jpg')
    # print('image shape', image.shape[0])
    height = image.shape[0]
    y_start_stop = [int(height*4//8), height]
    # print('ystart stop', y_start_stop)

    # draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    # image = image.astype(np.float32)/255

    # ystart = 400
    # ystop = 656
    scale = 1.5
    
    # out_img, bboxes = find_cars(image, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    # # plt.imshow(out_img)
    # # plt.title('boxes')
    # # plt.show()

    # zero_img = np.zeros_like(image[:, :, 0].astype(np.float))
    # # [:, :, 0]-- first : is first dimension, second : is second, 0 is only the r in rgb 3rd d
    # # plt.imshow(zero_img)
    # # plt.title('empty')
    # # plt.show()

    # heatmap = increment_heatmap(zero_img, bboxes)
    # # plt.imshow(heatmap)
    # # plt.title('heatmap')
    # # plt.show()

    # threshed_heat = apply_thresh(heatmap, 2)
    # # plt.imshow(threshed_heat)
    # # plt.title('threshed heatmap')
    # # plt.show()

    # # apply label() to get [heatmap_w/_labels, num_labels]
    # labels = label(threshed_heat)
    # # print("labels", labels[1])
    # labeled_image = box_labels(image, labels)
    # plt.imshow(labeled_image)
    # plt.title('with labeled cars')
    # plt.show()

    '''
    0. run video player on project video
    1. try to run model on just one image
    2. get class working
    4. every couple of frames, calculate new set
    '''
    boxes = Boxes()
    # print('after boxes')
    boxes.save_box('one')
    boxes.save_box('two')
    boxes.save_box('three')
    boxes.save_box('four')
    boxes.save_box('five')
    test = boxes.get_three()
    test.qsize()
    taken_val = test.get()
    print('taken val is: ', taken_val)
    test2 = boxes.get_three()
    test2.qsize()
    second_taken = test2.get()
    print('second taken is: ', second_taken)

    # boxed_cars = 'output.mp4'
    # clip = VideoFileClip('tet_video.mp4')
    

    # output_clip = clip.fl_image(test_func)
    # output_clip.write_videofile(boxed_cars, audio=False)
