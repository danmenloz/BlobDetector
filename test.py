import diptools as dip
import numpy as np
import cv2 as cv
import pdb

# README:
# Instructions
# Leave uncommented only the test section you want to run



# print("\MAGNITUDE AND PHASE PLOTS\n")
# img_file = './Images/lena.png'
# img = cv.imread(img_file,cv.IMREAD_GRAYSCALE)
# img_scld = dip.scaleImageChannel(img,1,np.float32) # scale image to range [0,1]
# F = dip.DFT2(img_scld) # compute DFT2
# F_shift = np.fft.fftshift(F)
# magnitude_spectrum = np.log(1+np.abs(F_shift)) # 1+np.abs(F_shift) is the log tranformation, just for visualization
# magnitude_spectrum = dip.scaleImageChannel(magnitude_spectrum,255,np.uint8)
# phase_angle = np.angle(F_shift)
# phase_angle = dip.scaleImageChannel(phase_angle,255,np.uint8)
# cv.imshow('Original Image',img)
# cv.imshow('Magnitude Spectrum',magnitude_spectrum)
# cv.imshow('Phase Angle',phase_angle)
# cv.waitKey()
# cv.destroyAllWindows()




print("\n FORWARD AND INVERSE DISCRETE FOURIER TRANSFORM \n")
img_file = './Images/lena.png'
img = cv.imread(img_file,cv.IMREAD_GRAYSCALE)
Fimg = dip.DFT2(img) # Fourier transform
fimg_c = dip.IDFT2(Fimg) # Inverse Fourier transform
fimg = fimg_c.astype(np.uint8) # Real part only and convert to uint
error = (img-fimg).astype(np.uint8) # compute error
print('Max error: ' + str(np.max(error)))
cv.imshow('Original Image',img)
cv.imshow('IDFT2 Result',fimg)
cv.imshow('IDFT2 Check',error)
cv.waitKey()
cv.destroyAllWindows()
