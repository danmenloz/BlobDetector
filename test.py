import diptools as dip
import numpy as np
import cv2 as cv
import pdb
import pyfftw # FFT library
import time

# README:
# Instructions
# Leave uncommented only the test section you want to run



print("\nMAGNITUDE AND PHASE PLOTS\n")
img_file = './Images/lena.png'
img = cv.imread(img_file,cv.IMREAD_GRAYSCALE)
# Call diptool function
[hmag, hphase] = dip.freqz(img, normalized=True, centered=True)
# plot responses
cv.imshow('Original Image',img)
cv.imshow('Magnitude Spectrum',hmag)
cv.imshow('Phase Angle',hphase)
cv.waitKey()
cv.destroyAllWindows()


# print("\n FORWARD AND INVERSE DFT DANIEL \n")
# start_time = time.time()
# img_file = './Images/lena.png'
# img = cv.imread(img_file,cv.IMREAD_GRAYSCALE)
# Fimg = dip.DFT2_PYFFTW(img) # Fourier transform
# fimg_c = dip.IDFT2__PYFFTW(Fimg) # Inverse Fourier transform
# fimg = fimg_c.astype(np.uint8) # Real part only and convert to uint
# error = (img-fimg).astype(np.uint8) # compute error
# print("    time(s): " + str(time.time()-start_time))
# print('Max error: ' + str(np.max(error)))
# cv.imshow('Original Image',img)
# cv.imshow('IDFT2 Result',fimg)
# cv.imshow('IDFT2 Check',error)
# cv.waitKey()
# cv.destroyAllWindows()
#
# print("\n FORWARD AND INVERSE DFT ALIETH \n")
# start_time = time.time()
# img_file = './Images/lena.png'
# img = cv.imread(img_file,cv.IMREAD_GRAYSCALE)
# Fimg = dip.DFT2(img) # Fourier transform
# fimg_c = dip.IDFT2(Fimg) # Inverse Fourier transform
# fimg = fimg_c.astype(np.uint8) # Real part only and convert to uint
# error = (img-fimg).astype(np.uint8) # compute error
# print("    time(s): " + str(time.time()-start_time))
# print('Max error: ' + str(np.max(error)))
# cv.imshow('Original Image',img)
# cv.imshow('IDFT2 Result',fimg)
# cv.imshow('IDFT2 Check',error)
# cv.waitKey()
# cv.destroyAllWindows()



# print("\n FORWARD AND INVERSE DFT RGB \n")
# start_time = time.time()
# img_file = './Images/lena.png'
# img = cv.imread(img_file)
#
# Fimg = np.empty(img.shape, dtype='complex128')
# fimg_c = np.empty(img.shape, dtype='complex128')
# fimg = np.empty(img.shape, dtype=np.uint8)
#
# # DFT of each channel
# for ch in range(img.shape[2]):
#         Fimg[:,:,ch] = dip.DFT2(img[:,:,ch]) # Fourier transform
#
# # iDFT of each channel
# for ch in range(img.shape[2]):
#         fimg_c[:,:,ch] = dip.IDFT2(Fimg[:,:,ch]) # Fourier transform
#         fimg[:,:,ch] = fimg_c[:,:,ch].astype(np.uint8) # Real part only and convert to uint
#
# error = (img-fimg).astype(np.uint8) # compute error
#
# print("    time(s): " + str(time.time()-start_time))
# print('Max error: ' + str(np.max(error)))
# cv.imshow('Original Image',img)
# cv.imshow('IDFT2 Result',fimg)
# cv.imshow('IDFT2 Check',error)
# cv.waitKey()
# cv.destroyAllWindows()




# print("\n DFT TIMING: fftw VS numpy \n")
# # Generate some data
# a = pyfftw.empty_aligned((1280, 640), dtype='complex128')
# ar, ai = np.random.randn(2, 1280, 640)
# a[:] = ar + 1j*ai
# # fftw DFT computation
# start_time = time.time()
# fft_object = pyfftw.builders.fft(a)
# b = fft_object()
# print("    fftw time(s): " + str(time.time()-start_time))
# # numpy DFT computation
# start_time = time.time()
# c = np.fft.fft(a)
# print("    numpy time(s): " + str(time.time()-start_time))
