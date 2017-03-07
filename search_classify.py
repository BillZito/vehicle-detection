import cv2
import glob
import time
import pickle
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
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

    return window_list
    # return draw_img, window_list

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
    final_boxes = []
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
        # avg_x = np.sum(nonzero_x) / nonzero_x.shape[0]
        # avg_y = np.sum(nonzero_y) / nonzero_y.shape[0]
        # print('avg x', avg_x, 'avg y', avg_y)
        # print('len x', nonzero_x.shape[0], 'leny', nonzero_y.shape[0])

        # left_points = []
        # right_points = []
        
        # for x_point in curr_points[1]:
        #     if x_point < avg_x:
        #         left_points.append(x_point)
        #     else: 
        #         right_points.append(x_point)

        # top_points = []
        # bottom_points = []
        # for y_point in curr_points[0]:
        #     if y_point < avg_y:
        #         bottom_points.append(y_point)
        #     else:
        #         top_points.append(y_point)

        # left_avg = np.sum(np.array(left_points)) // len(left_points)
        # right_avg = np.sum(np.array(right_points)) // len(right_points)
        # top_avg = np.sum(np.array(top_points)) // len(top_points)
        # bottom_avg = np.sum(np.array(bottom_points)) // len(bottom_points)
        # print('left', left_avg, 'bottom', bottom_avg, 'right', right_avg, 'top', top_avg, )
        # print('sum', np.sum(nonzero_x)/nonzero_x.shape[0])
        # try reversing top and bottom if doesnt work
        # bbox = ((left_avg, bottom_avg), (right_avg, top_avg))
        bbox = ((np.min(nonzero_x), np.min(nonzero_y)), (np.max(nonzero_x), np.max(nonzero_y)))
        final_boxes.append(bbox)
        # print('bbox', bbox[0], bbox[1])
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    return img, final_boxes

class Boxes:

    def __init__(self):
        self.max_3 = deque()
        self.found_5 = deque()
        self.labels = 0

    def save_box(self, box_list):
        self.max_3.append(box_list)
        # print('save box called, size now: ', len(self.max_3))
        if len(self.max_3) > 3:
            throw_away = self.max_3.popleft()
            # print('get called to remove: ', throw_away)

    def save_final(self, box_list):
        self.found_5.append(box_list)
        # print('save final called, size now', len(self.found_5))

        if len(self.found_5) > 8:
            throw_away = self.found_5.popleft()

    def set_labels(self, num):
        self.labels = num

    def get_labels(self):
        return self.labels

    def get_orig(self):
        return self.max_3

    def get_finals(self):
        return self.found_5

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
    image = mpimg.imread('test_images/test4.jpg')
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
    scale = 1
    
    '''
    2) determine which values correspond to the same car-- do another heatmap? 
    3) average/ extrapolate after 3-5 frames


    '''
    boxes = Boxes()
    # bboxes = find_cars(image, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    # boxes.save_box(bboxes)
    # arrs = boxes.get_orig()
    # print('arrs are ', arrs)

    # # plt.imshow(out_img)
    # # plt.title('boxes')
    # # plt.show()
    def process_image(image):
        img_copy = np.copy(image)
        bboxes = find_cars(image, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        print('hog found', len(bboxes))
        boxes.save_box(bboxes)
        arrs = boxes.get_orig()
        # print('length of boxes: ', len(arrs))
        # print('datatype', arrs[0])

        combo_box = []
        for box_list in arrs:
            combo_box += box_list
            # print('combo box now', combo_box)

        hog_zero_img = np.zeros_like(image[:, :, 0].astype(np.float))
        # # [:, :, 0]-- first : is first dimension, second : is second, 0 is only the r in rgb 3rd d
        # # plt.imshow(hog_zero_img)
        # # plt.title('empty')
        # # plt.show()

        hog_heatmap = increment_heatmap(hog_zero_img, combo_box)
        # plt.imshow(hog_heatmap)
        # plt.title('hog_heatmap')
        # plt.show()

        # print('length of arr', len(arrs))
        if len(bboxes) < 1 or len(arrs) < 2:
            hog_threshed_heat = apply_thresh(hog_heatmap, 0)
        elif len(arrs) < 3:
            hog_threshed_heat = apply_thresh(hog_heatmap, 2)
        else:  
            hog_threshed_heat = apply_thresh(hog_heatmap, 5)
        # plt.imshow(hog_threshed_heat)
        # plt.title('threshed heatmap')
        # plt.show()

        # # apply label() to get [heatmap_w/_labels, num_labels]
        hog_labels = label(hog_threshed_heat)
        # # print("hog_labels", hog_labels[1])
        hog_labeled_image, final_boxes = box_labels(image, hog_labels)
        # plt.imshow(hog_labeled_image)
        # plt.title('with labeled cars')
        # plt.show()
        boxes.save_final(final_boxes)

        final_arrs = boxes.get_finals()

        all_finals = []
        for box_list in final_arrs:
            all_finals += box_list

        final_zero_img = np.zeros_like(img_copy[:, :, 0].astype(np.float))
        # plt.imshow(final_zero_img)
        # plt.title('empty')
        # plt.show()

        final_heatmap = increment_heatmap(final_zero_img, all_finals)
        plt.imshow(final_heatmap)
        plt.title('final_heatmap')
        plt.show()

        print('len final arrs', len(final_arrs))
        if len(bboxes) < 1 or len(final_arrs) < 5:
            final_threshed_heat = apply_thresh(final_heatmap, 0)
        else: 
            final_threshed_heat = apply_thresh(final_heatmap, 3)
        plt.imshow(final_threshed_heat)
        plt.title('final threshed heatmap')
        plt.show()

        # # apply label() to get [heatmap_w/_labels, num_labels]
        final_labels = label(final_threshed_heat)

        # if less labels, make sure aren't filtering too much
        # print('final labels', final_labels[1], 'prev labels', boxes.get_labels())
        # if final_labels[1] < boxes.get_labels():
        #     # plt.imshow(final_heatmap)
        #     # plt.title('final_heatmap')
        #     # plt.show()

        #     final_threshed_heat = apply_thresh(final_heatmap, 0)
        #     final_labels = label(final_threshed_heat)
        #     print('after redo', final_labels[1])
        #     plt.imshow(final_threshed_heat)
        #     plt.title('after redo')
        #     plt.show()

        boxes.set_labels(final_labels[1])
        # # print("final_labels", final_labels[1])
        final_labeled_image, last_boxes = box_labels(img_copy, final_labels)
        plt.imshow(final_labeled_image)
        plt.title('with final perform')
        plt.show()

        return final_labeled_image
        # return hog_labeled_image
    # process_image(image)


    # boxed_cars_vid = 'project_output.mp4'
    # clip = VideoFileClip('project_video.mp4')
    
    # boxed_cars_vid = 'test_output3.mp4'
    # clip = VideoFileClip('test_video.mp4')

    boxed_cars_vid = 'short_pass_output.mp4'
    clip = VideoFileClip('short_pass_vid.mp4')

    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(boxed_cars_vid, audio=False)
