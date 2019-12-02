# Scale Space using Difference of Gaussian

import diptools as dip
import numpy as np
import cv2 as cv
import pdb
import time
import math

# Read image
img_file = './Images/butterfly.jpg'
img = cv.imread(img_file,cv.IMREAD_GRAYSCALE)

# DoG Parameters
s = 2 # octave subdivision
k = 2**(1/s) # sigma factor
num_imgs = s+3 # number of images in the octave
ksize = 11 # Gaussian kernel size
sigma = 1.6 # standard deviation for Gaussian kernel

# Generate octave
scl_space = []
for scl in range(num_imgs):
    G = dip.GaussianKernel(ksize, (k**scl)*sigma)
    # L = dip.conv2(img, G,  dip.Pad.REFLECT_ACROSS_EDGE)
    L = dip.fftconv2(img, G)
    scl_space.append(L.astype(np.uint8)) # add Blurred image to the octave

# # Show octave
# for i in range(len(scl_space)):
#     cv.imshow('L_'+str(i+1),scl_space[i])
# print('Done!')
# cv.waitKey()
# cv.destroyAllWindows()

# compute DoG
DoG = []
for scl in range(num_imgs-1):
    D = scl_space[scl+1] - scl_space[scl]
    DoG.append(D)

# Show DoG space
for i in range(len(DoG)):
    # pdb.set_trace()
    cv.imshow('D_'+str(i+1), dip.scaleImage(DoG[i], modo='custom') )
print('Done!')
cv.waitKey()
cv.destroyAllWindows()

# DoG_filtered = []
# E = [] # Extrema points list
# for i in range(1,len(DoG)-1):
#     D = DoG[i]
#     for y in range(D.shape[0]):
#         for x in range(D.shape[1]):
#             # pdb.set_trace()
#             N = dip.DoGNeighbors((DoG[i-1],DoG[i],DoG[i+1]), (x,y))
#             # check maximum
#             if D[y][x] > max(N) or D[y][x] < min(N):
#                 E.append((x,y)) # save point
