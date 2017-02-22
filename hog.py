'''
get all of the hog features from lower half of the image
TODO: apply to lower half
'''
from skimage.feature import hog
orient = 0
pix_per_cell = 8
cell_per_block = 2


feature_array = hog(img, orientations=orient, \
  pixels_per_cell=(pix_per_cell, pix_per_cell),\
  cells_per_block=(cell_per_block, cell_per_block),\
  visualize=False, feature_vector=False)
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

