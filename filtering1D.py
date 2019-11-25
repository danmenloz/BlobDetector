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
img_shifted = np.empty((P,Q), np.int8)
for y in range(P):
    for x in range(Q):
        img_shifted[y][x] = img_cropped[y][x] * (-1)**(x+y)
[hmag, hphase] = dip.freqz(img_shifted, normalized=True, centered=False)

# Step 4
# Compute DFT
# Fimg = np.empty((P,Q,3), dtype=complex)
# for ch in range(nchls):
#     Fimg[:,:,ch] = dip.DFT2(img_shifted[:,:,ch]) # Fourier transform
#     print("Fimg min,max: " + str(np.min(Fimg[:,:,ch])) + ", " + str(np.max(Fimg[:,:,ch])))

# # Step 5
# # real-symmetric transfer funtion H(u,v) of size PxQ centerd at (P/2,Q,2)
# sigma = 2 # try different values
# ksize = max(P, Q)
# # pdb.set_trace()
# G = cv.getGaussianKernel(ksize,sigma)
# H = np.multiply(G,np.transpose(G))
# # pdb.set_trace()
# if Q>P:
#     ha = int((Q-P)/2)
#     H = H[ha:ha+P,:]
# elif Q<P:
#     hb = int((P-Q)/2)
#     H = H[:,hb:hb+Q]
#     H = np.asarray(H, dtype=np.double)
# print("H Kernel: " + str(H.shape))
# # pdb.set_trace()
# print("H min,max: " + str(np.min(H)) + ", " + str(np.max(H)) )
#
# # Step 6
# # H(u,v)*F(u,v)
# Gimg = np.empty((P,Q,3), dtype=complex)
# for ch in range(nchls):
#     Gimg[:,:,ch] = np.multiply(Fimg[:,:,ch], H)
#     print("Gimg min,max: " + str(np.min(Gimg[:,:,ch])) + ", " + str(np.max(Gimg[:,:,ch])))
# print("Gimg: " + str(Gimg.shape))
#
# # Step 7
# # Filtered image through iDFT
# gimg = np.empty((P,Q,3), np.double)
# g_padded = np.empty((P,Q,3), np.int8)
# for ch in range(nchls):
#     gimg[:,:,ch] = (dip.IDFT2(Gimg[:,:,ch])).real
#     print("gimg min,max: " + str(np.min(gimg[:,:,ch])) + ", " + str(np.max(gimg[:,:,ch])))
#
# for ch in range(nchls):
#     for y in range(P):
#         for x in range(Q):
#             g_padded[y,x,ch] = (gimg[y,x,ch] * (-1)**(x+y)).astype(np.int8)
#     print("g_padded min,max: " + str(np.min(g_padded[:,:,ch])) + ", " + str(np.max(g_padded[:,:,ch])))
# print("gimg: " + str(g_padded.shape))


cv.imshow('1 Original', img)
cv.imshow('2 Padding',img_cropped)
cv.imshow('3 Shifted', dip.scaleImage(img_shifted, crop=True))
# cv.imshow('Gpadded',dip.scaleImage(g_padded, modo = 'custom', K = 255))
cv.imshow('Mag Spectrum img_shifted', hmag)
cv.waitKey()
cv.destroyAllWindows()
