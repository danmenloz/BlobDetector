# Filtering in frequency domain
# Steps described in Ch 4, pag 266, DIP book
import diptools as dip
import numpy as np
import cv2 as cv
import pdb
import pyfftw # FFT library
import time


img_file = './Images/sunflowers.jpg'
img = cv.imread(img_file)
nrows = img.shape[0]
ncols = img.shape[1]
nchls = img.shape[2]
print("img: " + str(img.shape))

# Step 1
# padded sizes P Q
P = nrows*2
Q = ncols*2
print("P,Q: " + str(P) + ", " + str(Q))

# Step 2
# form padded image
img_padded = dip.padding(img, dip.Pad.REFLECT_ACROSS_EDGE, mn=(nrows+1,ncols+1) ) # +1 because of the shared border with kernel
# crop padded image
img_cropped = img_padded[nrows:,ncols:]
print("img_cropped: " + str(img_cropped.shape))

# Step 3
# multiply by (-1)^(x+y)
img_shifted = np.empty(img.shape, np.int8)
for ch in range(nchls):
    for y in range(nrows):
        for x in range(ncols):
            img_shifted[y][x][ch] = img_cropped[y][x][ch] * (-1)**(x+y)

# Step 4
# Compute DFT
Fimg = dip.DFT2(img_shifted) # Fourier transform

# Step 5
# real-symmetric transfer funtion H(u,v) of size PxQ centerd at (P/2,Q,2)
sigma = 2 # try different values
ksize = max(P, Q)
# pdb.set_trace()
G = cv.getGaussianKernel(ksize,sigma)
W = np.multiply(G,np.transpose(G))
# pdb.set_trace()
if Q>P:
    wa = int((Q-P)/2)
    W = W[wa:wa+P,:]
elif Q<P:
    wb = int((P-Q)/2)
    W = W[:,wb:wb+Q]
print("G Kernel: " + str(W.shape))






cv.imshow('Padding',img_cropped)
cv.imshow('Shifted', dip.scaleImage(img, modo = 'custom', K = 255))
cv.waitKey()
cv.destroyAllWindows()
