# Filtering in frequency domain
# Steps described in Ch 4, pag 266, DIP book
import diptools as dip
import numpy as np
import cv2 as cv
import pdb
import pyfftw # FFT library
import time


img_file = './Images/lena.png'
img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
nrows = img.shape[0]
ncols = img.shape[1]
print("img: " + str(img.shape))

# Step 1
# padded sizes P Q
P = nrows*2
Q = ncols*2
print("P,Q: " + str(P) + ", " + str(Q))

# Step 2
# form padded image
img_padded = dip.padding(img, dip.Pad.CLIP_ZERO, mn=(nrows+1,ncols+1) ) # +1 because of the shared border with kernel
# crop padded image
img_cropped = img_padded[nrows:,ncols:]
print("img_cropped: " + str(img_cropped.shape))

# Step 3
# multiply by (-1)^(x+y)
img_shifted = np.empty((P,Q), np.int)
for y in range(P):
    for x in range(Q):
        img_shifted[y][x] = img_cropped[y][x] * (-1)**(x+y)
print("img_shifted min,max: " + str(np.min(img_shifted)) + ", " + str(np.max(img_shifted)))
[hmag, hphase] = dip.freqz(img_shifted, normalized=True, centered=False)

# Step 4
# Compute DFT
Fimg = np.empty((P,Q), dtype=complex)
Fimg= dip.DFT2(img_shifted) # Fourier transform
print("Fimg min,max: " + str(np.min(Fimg)) + ", " + str(np.max(Fimg)))

# Step 5
# real-symmetric transfer funtion H(u,v) of size PxQ centerd at (P/2,Q,2)
sigma = 10 # try different values
ksize = max(P, Q)
# pdb.set_trace()
# G = cv.getGaussianKernel(ksize,sigma)
# H = np.multiply(G,np.transpose(G))
H = np.ones((ksize,ksize))
# pdb.set_trace()
if Q>P:
    ha = int((Q-P)/2)
    H = H[ha:ha+P,:]
elif Q<P:
    hb = int((P-Q)/2)
    H = H[:,hb:hb+Q]
    H = np.asarray(H, dtype=np.double)
print("H Kernel: " + str(H.shape))
# pdb.set_trace()
print("H min,max: " + str(np.min(H)) + ", " + str(np.max(H)) )

# Step 6
# H(u,v)*F(u,v)
Gimg = np.empty((P,Q), dtype=complex)
Gimg = np.multiply(Fimg, H)
print("Gimg min,max: " + str(np.min(Gimg)) + ", " + str(np.max(Gimg)))
print("Gimg: " + str(Gimg.shape))

# Step 7
# Filtered image through iDFT
gimg = np.empty((P,Q), np.double)
g_padded = np.empty((P,Q), np.double)
gimg= (dip.IDFT2(Fimg)).real # changed Gimg for Fimg to test
print("gimg min,max: " + str(np.min(gimg)) + ", " + str(np.max(gimg)))
[hmag1, hphase1] = dip.freqz(gimg, normalized=True, centered=False)

for y in range(P):
    for x in range(Q):
        g_padded[y][x] = gimg[y][x] * (-1)**(x+y)   #).astype(np.int8)
print("g_padded min,max: " + str(np.min(g_padded)) + ", " + str(np.max(g_padded)))
print("gimg: " + str(g_padded.shape))


cv.imshow('1 Original', img)
cv.imshow('2 Padding',img_cropped)
cv.imshow('3 Shifted', dip.scaleImage(img_shifted, crop=True))
cv.imshow('4 Mag Spectrum img_shifted', hmag)
cv.imshow('5 Gaussian Filter', dip.scaleImage(H,modo='custom',K=255))
cv.imshow('6 Mag Spectrum H*F', hmag1)
cv.imshow('7 g padded', dip.scaleImage(g_padded,modo='custom',K=255))
print('Done!')
cv.waitKey()
cv.destroyAllWindows()
