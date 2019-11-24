# Project 04 Digital Imaging Processing
# Authors:
#   * Daniel Mendez (dmendez@ncsu.edu)
#   * Jaqueline
#   * Alieth
# Version: python 3.7.4
# Description: This file contains the core functions developed in all previous
#   projects.

from enum import Enum
import numpy as np
import cv2 as cv
import pyfftw # FFT library
import pdb

 # Padding type
class Pad(Enum):
    CLIP_ZERO = 0
    WRAP_AROUND = 1
    COPY_EDGE = 2
    REFLECT_ACROSS_EDGE = 3

# Predifined 2D kernels
class Kernel(Enum):
    BOX_FILTER = 0
    FIRST_ORDER_DERIVATIVE_1 = 1
    FIRST_ORDER_DERIVATIVE_2 = 2
    FIRST_ORDER_DERIVATIVE_3 = 3
    PREWIT_Z = 4
    PREWIT_Y = 5
    SOBEL_X = 6
    SOBEL_Y = 7
    ROBERTS_X = 8
    ROBERTS_Y = 9
    LAPLACIAN_1 = 10
    LAPLACIAN_2 = 11



# Convolution in 2D
# conv2(f,w,pad)
# f     - input image (grey or RGB)
# w     - 2D kernel (check Kerenl Enum)
# pad   - 4 padding type (check Pad Enum)
def conv2(f,w,pad):
    # Check data types
    if not isinstance(f, np.ndarray):
        raise TypeError("arg 'f' must a image of type 'np.ndarray'")
        exit()
    if isinstance(w, Kernel) or isinstance(w, int):
        w = Kernel(w)
        if w == Kernel.BOX_FILTER:
            w = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
        elif w == Kernel.FIRST_ORDER_DERIVATIVE_1:
            w = np.array([-1,1])
        elif w == Kernel.FIRST_ORDER_DERIVATIVE_2:
            w = np.array([[-1], [1]])
        elif w == Kernel.FIRST_ORDER_DERIVATIVE_3:
            w = np.array([[1], [-1]])
        elif w == Kernel.PREWIT_Z:
            w = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
        elif w == Kernel.PREWIT_Y:
            w = np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]])
        elif w == Kernel.SOBEL_X:
            w = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, -1]])
        elif w == Kernel.SOBEL_Y:
            w = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
        elif w == Kernel.ROBERTS_X:
            w = np.array([[0, 1],[-1, 0]])
        elif w == Kernel.ROBERTS_Y:
            w = np.array([[1, 0],[0, -1]])
        elif w == Kernel.LAPLACIAN_1:
            w = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
        elif w == Kernel.LAPLACIAN_2:
            w = np.array([[0, 1, 0],[1, -8, 1],[0, 1, 0]])
    elif isinstance(w,np.ndarray):
        pass # do nothing
    else:
        raise TypeError("arg 'w' must be a valid Enum, 'int', or 'np.ndarray' ")
        exit()
    if isinstance(pad, Pad) or isinstance(pad, int):
        pad = Pad(pad)
    else:
        raise TypeError("arg 'pad' must be a valid Enum or 'int' ")
        exit()

    # Perfom Padding
    # pdb.set_trace()
    img_padded = padding(f, pad, w.shape)
    # pdb.set_trace()

    # Get original image dimensions
    N = f.shape[1]
    M = f.shape[0]

    # Get kernel dimensions
    if len(w.shape)==2:
        m, n = w.shape
    elif len(w.shape) == 1:
        n = w.shape[0]
        m = 1

    # Check odd square size for x
    #  and set the center of w
    xc, yc = (0,0) # defualt center in kernel
    if n%2: # is n odd?
        xc = int(n/2)
    if m%2: # is m odd?
        yc = int(m/2)

    # Check image channels
    if len(f.shape)>2:
        # It is a color image
        img_channels = f.shape[2] #channels
        img_cvtd = np.zeros((img_padded.shape[0], img_padded.shape[1],img_channels))
    else:
        img_channels = 0
        img_cvtd = np.zeros((img_padded.shape[0], img_padded.shape[1]))

    # Peform Correlation operation
    if img_channels>0:
        for ch in range(img_channels):
            # peform Convolution operation
            for y in range(m+M-1): #y coordinate
                for x in range(n+N-1): # x coordinate
                    if m==1:
                        for wx in range(n):
                            img_cvtd[y+yc][x+xc][ch] += img_padded[y][x+wx][ch]*w[0][wx] # changed w[wx] ->  w[0][wx]
                    else:
                        for wy in range(m):
                            for wx in range(n):
                                img_cvtd[y+yc][x+xc][ch] += img_padded[y+wy][x+wx][ch]*w[wy][wx]
    else: # Grayscale image
        # peform Convolution operation
        for y in range(m+M-1): #y coordinate
            for x in range(n+N-1): # x coordinate
                if m==1:
                    # pdb.set_trace()
                    for wx in range(n):
                        img_cvtd[y+yc][x+xc] += img_padded[y][x+wx]*w[0][wx] # changed w[wx] ->  w[0][wx]
                else:
                    for wy in range(m):
                        for wx in range(n):
                            img_cvtd[y+yc][x+xc] += img_padded[y+wy][x+wx]*w[wy][wx]

    # resize scaled image to match the original image size
    # Useful variables
    ax, bx, cx, dx = (n-1, n-1+N-1, n-1, n-1+N-1)
    ay, by, cy, dy = (m-1, m-1, m-1+M-1, m-1+M-1)
    img_resized = img_cvtd[ay:cy+1, ax:bx+1]

    return img_resized # this is a float-value matrix, not suitable for display, use scaleImage after it




# Scale Image for display
def scaleImage(img, modo = 'auto', K = 255):
    # Check image channels
    if len(img.shape)>2:
        # It is a color image
        img_channels = img.shape[2] #channels
    else:
        img_channels = 0

    if modo == 'auto':
        # Scale image in range [0-img.max()]  modo Auto
        if img_channels>0:
            img_scld = np.zeros((img.shape[0],img.shape[1],img_channels),np.uint8)
            for ch in range(img_channels):
                img_scld[:,:,ch] = scaleImageChannel(img[:,:,ch],int(img[:,:,ch].max()),np.uint8) # try no to alter colors
        else:
            img_scld = scaleImageChannel(img,int(img.max()),np.uint8)
    elif modo == 'custom':
        # Scale image in range [0-img.max()]  modo Auto
        if img_channels>0:
            img_scld = np.zeros((img.shape[0],img.shape[1],img_channels),np.uint8)
            for ch in range(img_channels):
                img_scld[:,:,ch] = scaleImageChannel(img[:,:,ch],K,np.uint8)
        else:
            img_scld = scaleImageChannel(img,K,np.uint8)
    else:
        raise TypeError("invalid 'modo' argument")
        exit()

    return img_scld


# Scaled image with spanned the intensities in the range [0,K]
# scaleImageChannel(g,K,data_type)
# g         - 1D image
# K         - maximum value
# data_type - np data type for the result
def scaleImageChannel(g,K,data_type):
    # see page 91 of DIP book for reference
    min_g = g.min()

    # ensure K max value for 8 bits
    if K > 255:
        K = 255

    for y in range(g.shape[0]): #y coordinate
    	for x in range(g.shape[1]): # x coordinate
            g[y][x] -= min_g

    g_s = np.empty((g.shape[0],g.shape[1]),data_type)
    max_g = g.max()

    for y in range(g_s.shape[0]): #y coordinate
    	for x in range(g_s.shape[1]): # x coordinate
            g_s[y][x] = data_type(K*(g[y][x]/max_g))

    return g_s




# Padding function
# padding(img, pad, mn=(1,1) )
# img - Image
# pad - enumeration Pad
# mn - tuple indicating the dimesion of the kernel
# xy - origin of the kernel, default (0,0)
def padding(img, pad, mn=(1,1) ):
    # Check kernel size
    if len(mn)==2:
        m, n = mn
    elif len(mn) == 1:
        n = mn[0]
        m = 1
    # pdb.set_trace()

    img_width = img.shape[1]
    img_height = img.shape[0]

    # Check image channels
    if len(img.shape)>2:
        # It is a color image
        img_channels = img.shape[2] #channels
        img_padded = np.zeros((img_height+2*(m-1), img_width+2*(n-1),img_channels), img.dtype)
    else:
        img_channels = 0
        img_padded = np.zeros((img_height+2*(m-1), img_width+2*(n-1)), img.dtype)

    # Check image size
    new_height, new_width = (img_padded.shape[0], img_padded.shape[1])

    # Copy image in shifted position
    for y in range(img_height): #y coordinate
        for x in range(img_width): # x coordinate
            # pdb.set_trace()
            img_padded[y+m-1][x+n-1]= img[y][x]

    # Useful variables
    M = img_height
    N = img_width
    ax, bx, cx, dx = (n-1, n-1+N-1, n-1, n-1+N-1)
    ay, by, cy, dy = (m-1, m-1, m-1+M-1, m-1+M-1)

    # if img_channels==0: # Grayscale images
    if pad == Pad.CLIP_ZERO:
        # Already padded with 0's
        return img_padded
    elif pad == Pad.WRAP_AROUND:
        for x in range(new_width): # x coordinate
            if x<(n-1):
                img_padded[m-1:m-1+img_height,x:x+1] = img[0:img_height,N-n+1+x:N-n+1+(x+1)]
            if x>(n-1+img_width-1):
                img_padded[m-1:m-1+img_height,x:x+1] = img[0:img_height,x-(n-1+img_width):x+1-(n-1+img_width)]
        for y in range(new_height): #y coordinate
            if y<(m-1):
                img_padded[y:y+1,:] = img_padded[cy-(m-1)+y-1:cy-(m-1)+y,: ]
            if y>(m-1+img_height-1):
                img_padded[y:y+1,:] = img_padded[m+(y-(m+img_height)):m+1+(y-(m+img_height)),:]
        return img_padded
    elif pad == Pad.COPY_EDGE:
        for x in range(new_width): # x coordinate
            if x<(n-1):
                img_padded[m-1:m-1+img_height,x:x+1] = img[:,0:1]
            if x>(n-1+img_width-1):
                img_padded[m-1:m-1+img_height,x:x+1] = img[:,img_width-1:img_width]
        for y in range(new_height): #y coordinate
            if y<(m-1):
                img_padded[y:y+1,:] = img_padded[m-1:m,: ]
            if y>(m-1+img_height-1):
                img_padded[y:y+1,:] = img_padded[m-1+img_height-1:m-1+img_height,:]
        return img_padded
    elif pad == Pad.REFLECT_ACROSS_EDGE:
        for x in range(new_width): # x coordinate
            if x<(n-1):
                img_padded[m-1:m-1+img_height,x:x+1] = img[:,n-2-x:n-1-x]
            if x>(n-1+img_width-1):
                img_padded[m-1:m-1+img_height,x:x+1] = img[:,img_width-1-(x-(n-1+img_width)):img_width-(x-(n-1+img_width))]
        for y in range(new_height): #y coordinate
            if y<(m-1):
                img_padded[y:y+1,:] = img_padded[ay+(m-1)-1-y:ay+(m-1)-y,: ] # img_padded[:,m-y:m+1-y ]
            if y>(m-1+img_height-1):
                img_padded[y:y+1,:] = img_padded[m-1+img_height-1-(y-(m-1+img_height)):m-1+img_height-(y-(m-1+img_height)),:]
        return img_padded




# Fast Fourier Transform 2D
# f - Grayscale input image
def DFT2(f):
    # Initialize input and outpu vectors
    a = pyfftw.empty_aligned(f.shape[1], dtype='complex128')
    b = pyfftw.empty_aligned(f.shape[1], dtype='complex128')

    fft_row = pyfftw.FFTW(a,b) # create FFTW object

    # Create an empty array to store the FFT
    Fxv = np.empty((f.shape[0],f.shape[1]), dtype='complex128')

    # Perfom FFT on all rows of f
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):
            a[x] = f[y][x] # copy row in 'a' array
        Fxv[y,:] = fft_row() # perfrom FFT and store it in array 'Fxv'

    # Initialize input and outpu vectors
    c = pyfftw.empty_aligned(f.shape[0], dtype='complex128')
    d = pyfftw.empty_aligned(f.shape[0], dtype='complex128')

    fft_col = pyfftw.FFTW(c,d) # create FFTW object

    Fuv = np.empty((f.shape[0],f.shape[1]), dtype='complex128')

    # Perfom FFT on all columns of Fuv
    for x in range(Fuv.shape[1]):
        for y in range(Fuv.shape[0]):
            c[y] = Fxv[y][x] # copy row in 'a' array
        Fuv[:,x] = fft_col() # perfrom FFT and store it in array 'Fuv'

    return Fuv



# Inverse Fast Fourier Transform 2D
# F - input Grayscale image
def IDFT2(F):
    # Create emtpy matrix to store the 2D IDFT
    Fu = np.empty((F.shape[0],F.shape[1]), dtype='complex128')

    # swap F
    for y in range(F.shape[0]):
        for x in range(F.shape[1]):
            re = F[y][x].real
            im = F[y][x].imag
            Fu[y][x] = im + 1j*re

    # compute 2D-DFT
    Fx = DFT2(Fu)

    # swap Fx
    for y in range(Fx.shape[0]):
        for x in range(Fx.shape[1]):
            re = Fx[y][x].real
            im = Fx[y][x].imag
            Fx[y][x] = im + 1j*re

    return Fx*(1/(F.shape[0]*F.shape[1]))




# main function
def main():
    # Print module information
    print('''Module diptools.py
    Functions in this module:
    - conv2(f,w,pad)
    - scaleImageChannel(g,K,data_type)
    - padding(img, pad, mn=(1,1))
    - DFT2(f)
    - IDFT2(F)
    Enumerations in this module:
    - Pad(Enum)
        CLIP_ZERO
        WRAP_AROUND
        COPY_EDGE
        REFLECT_ACROSS_EDGE
    - Kernel(Enum):
        BOX_FILTER
        FIRST_ORDER_DERIVATIVE_1
        FIRST_ORDER_DERIVATIVE_2
        FIRST_ORDER_DERIVATIVE_3
        PREWIT_Z
        PREWIT_Y
        SOBEL_X
        SOBEL_Y
        ROBERTS_X
        ROBERTS_Y
        ''')

# Call main function
if __name__ == "__main__":
    main()
