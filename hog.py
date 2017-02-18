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
TODO: use sliding windows to find which hog features are in
which window
'''