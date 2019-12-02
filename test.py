import diptools as dip
import numpy as np
import cv2 as cv
import pdb
import pyfftw # FFT library
import time
import math

# README:
# Instructions
# Leave uncommented only the test section you want to run


# print("\nMAGNITUDE AND PHASE PLOTS\n")
# img_file = './Images/lena.png'
# img = cv.imread(img_file,cv.IMREAD_GRAYSCALE)
# # Call diptool function
# [hmag, hphase] = dip.freqz(img, normalized=True, centered=True)
# # plot responses
# cv.imshow('Original Image',img)
# cv.imshow('Magnitude Spectrum',hmag)
# cv.imshow('Phase Angle',hphase)
# cv.waitKey()
# cv.destroyAllWindows()


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



# print("\nGAUSSIAN KERNEL GENERATION\n")
# ksize = 3
# sigma = 3
# # diptools implementation
# G1 = dip.GaussianKernel(ksize, sigma)
# print('G1 Max: ' + str(np.max(G1)))
# print('G1 Min: ' + str(np.min(G1)))
# print('G1: ' + str(G1))
# print(G1.dtype)
# # mathematical implementation
# G2 = np.empty((ksize, ksize),np.float64)
# for y in range(ksize):
#     for x in range(ksize):
#         G2[y][x] = (1/(2*math.pi*sigma**2)) * np.exp(-((x-1)**2+(y-1)**2)/(2*sigma**2)) #
# # pdb.set_trace()
# print('G2 Max: ' + str(np.max(G2)))
# print('G2 Min: ' + str(np.min(G2)))
# print('G2: ' + str(G2))




# print("\nIMAGE BLURR BY CONVOLUTION\n")
# img_file = './Images/lena.png'
# img = cv.imread(img_file,cv.IMREAD_GRAYSCALE)
#
# # Blurr image using GuassianVector function
# g_vec = dip.GaussianVector((img.shape[0],img.shape[1]), 2)
# fltrd_part =  dip.conv2(img, g_vec,  dip.Pad.REFLECT_ACROSS_EDGE )
# fltrd_img1 =  dip.conv2(fltrd_part, g_vec.transpose(),  dip.Pad.REFLECT_ACROSS_EDGE )
#
# fltrd_img1 = fltrd_img1.astype(img.dtype)  # convert to np.uint image
#
# print('Max1: ' + str(np.max(fltrd_img1)))
# print('Min1: ' + str(np.min(fltrd_img1)))
#
# # Blurr image using GaussianKernel function
# s = 2 # octave subdivision
# ksize = 5
# sigma = 1.6
# k = 2**(1/s) # sigma factor
# G = dip.GaussianKernel(ksize, (k**4)*sigma)
# L = dip.conv2(img, G,  dip.Pad.REFLECT_ACROSS_EDGE)
# L = L.astype(img.dtype)
# print(img.dtype)
# print('Max2: ' + str(np.max(L)))
# print('Min2: ' + str(np.min(L)))
#
#
# cv.imshow('filtered', fltrd_img1)
# cv.imshow('original', img)
# cv.imshow('L', L)
# print('Done!')
# cv.waitKey()
# cv.destroyAllWindows()


x = 5
y = 5
# local neighbor indexes
il = [ [y-1,x-1], [y-1,x], [y-1,x+1], \
        [y,x-1],            [y,x+1],  \
       [y+1,x-1], [y+1,x], [y+1,x+1] ]


print(il)
