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
import time
import math

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
def scaleImage(img, modo = 'auto', K = 255, crop=False,):
    # Check image channels
    if len(img.shape)>2:
        # It is a color image
        img_channels = img.shape[2] #channels
    else:
        img_channels = 0

    if crop:
        # negative values are cropped to 0
        if img_channels>0:
            for ch in range(img_channels):
                for y in range(img.shape[0]): #y coordinate
                    for x in range(img.shape[1]): # x coordinate
                        if img[y][x][ch] < 0:
                            img[y][x][ch] = 0
        else:
            for y in range(img.shape[0]): #y coordinate
            	for x in range(img.shape[1]): # x coordinate
                    if img[y][x] < 0:
                        img[y][x] = 0

    if modo == 'auto':
        # Scale image in range [0-img.max()]  modo Auto
        if img_channels>0:
            img_scld = np.zeros((img.shape[0],img.shape[1],img_channels),np.uint8)
            for ch in range(img_channels):
                img_scld[:,:,ch] = scaleImageChannel(img[:,:,ch],int(img[:,:,ch].max()),np.uint8) # try no to alter colors
        else:
            img_scld = scaleImageChannel(img,int(img.max()),np.uint8)
    elif modo == 'custom':
        # Scale image in range [0-K]  modo Auto
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
    g_aux = g.astype(int) # copy of g as int to avoid overflow

    # ensure K max value for 8 bits
    if K > 255:
        K = 255

    for y in range(g.shape[0]): #y coordinate
    	for x in range(g.shape[1]): # x coordinate
            g_aux[y][x] -= min_g

    g_s = np.empty((g.shape[0],g.shape[1]),data_type)
    max_g = g_aux.max()

    for y in range(g_s.shape[0]): #y coordinate
    	for x in range(g_s.shape[1]): # x coordinate
            g_s[y][x] = data_type(K*(g_aux[y][x]/max_g))

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
def DFT2_PYFFTW(f):
    # Initialize input and output vectors
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

    # Initialize input and output vectors
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
def IDFT2__PYFFTW(F):
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




# Fast Fourier Transform 2D
# f - Grayscale input image
def DFT2(f):
    f = np.array(f)
    # FFT in Rows
    row_f = np.fft.fft(f)
    # FFT in Columns
    row_f = np.transpose(row_f)
    col_f = np.fft.fft(row_f)
    dft_2 = np.transpose(col_f)
    return dft_2




# Inverse Fast Fourier Transform 2D
# F - input Grayscale image
def IDFT2(f):
    f = np.array(f)
    # Calculating input matrix size
    size_x = f.shape[0]
    size_y = f.shape[1]
    # Conjugate input matrix
    conj_f = np.conj(f)
    # Calculating inverse fft
    idft_2 = 1/(size_x*size_y)*DFT2(conj_f)
    # return idft_2.real
    return idft_2




# Pyramid blending Function
# Arguments
# input_image - input image gray or RGB
# num_layers - number of layers of the pyramid
# Output
# gPyr - Gaussian Pyramid
# lPyr - Lapalacian Pyramid
# pyramid - select pyramids to compute: both, gaussian, laplacian
def ComputePyr(input_image, num_layers, pyramid='both'):

    img_height = input_image.shape[0]
    img_width = input_image.shape[1]
    [gPyr, lPyr] = [None,None]

    # num_layers must be positive
    if num_layers < 1:
        raise TypeError("Argument 'num_layers' must be a positive integer")
        exit()

    # check if num_layers is feasable
    # num of layers is inclusive, i.e. counting the original image
    h,w = [img_height, img_width]
    for i in range(1,num_layers-1):
        h,w = [int(h/2), int(w/2)]
        if h==1 or w==1:
            num_layers = i+1
            print('Warning: Maximum number of layers is: '+str(num_layers))
            break # exit for loop

    # Gaussian pyramid alwas has to be computed
    # generate Gaussian pyramid
    print("  Gaussian layer 1")
    gImg = input_image.copy()
    gPyr = [gImg] # first  element
    for i in range(num_layers-1):
        # gImg = cv.pyrDown(gImg)
        scale = (i+1)**2
        gImg = downSample(gImg, [int(gImg.shape[0]/2),int(gImg.shape[1]/2)], scale)
        print("  Gaussian layer " + str(i+2))
        gPyr.append(gImg)

    if pyramid=='both' or pyramid=='laplacian': # pyramid=='gaussian':
        #generate Laplacian Pyramid
        print("  Lapacian layer " + str(num_layers))
        lPyr = [gPyr[-1]] # assign first element as the last element of gauss_pyramid
        for i in range(num_layers-1, 0, -1): # for loop i reverse count
            scale = (i)**2
            gExt = upSample(gPyr[i], [gPyr[i-1].shape[0],gPyr[i-1].shape[1]], scale)
            print("  Lapacian layer " + str(i))
            lvl = cv.subtract(gPyr[i-1], gExt) # compute laplacian pyramid level
            lPyr.append(lvl)

    gPyr.reverse()
    return [gPyr, lPyr] # return list



# Nearest Neighbor Interpolation
# A - np.array image
# new_size - list [height,width]
def nn_interpolate(A, new_size):
    old_size = A.shape
    row_ratio, col_ratio = np.array(new_size)/np.array(old_size)

    # row wise interpolation
    row_idx = (np.ceil(range(1, 1 + int(old_size[0]*row_ratio))/row_ratio) - 1).astype(int)

    # column wise interpolation
    col_idx = (np.ceil(range(1, 1 + int(old_size[1]*col_ratio))/col_ratio) - 1).astype(int)

    final_matrix = A[:, col_idx][row_idx, :] #modified

    return final_matrix



# Upsample
# img - np image array
# new_size - size [height, width] of the output image
# scale - downscale or upscale as a power of 2: 2,4,6...
def upSample(img, new_size, scale):
    up_img = np.empty((new_size[0],new_size[1], img.shape[2]),img.dtype) # create temporal image

    # Upsample image
    for c in range(img.shape[2]): # channel
        up_img[:,:,c] = nn_interpolate(img[:,:,c],new_size)

    # Smooth image
    # g_vec = GaussianVector2(scale)
    g_vec = GaussianVector((up_img.shape[0],up_img.shape[1]), 2)
    fltrd_part =  conv2(up_img, g_vec,  Pad.REFLECT_ACROSS_EDGE )
    fltrd_img =  conv2(fltrd_part, g_vec.transpose(),  Pad.REFLECT_ACROSS_EDGE )

    return fltrd_img.astype(img.dtype)  # convert to np.uint image



# Downsample
# img - np image array
# new_size - size [height, width] of the output image
# scale - downscale or upscale as a power of 2: 2,4,6...
def downSample(img, new_size, scale):
    down_img = np.empty((new_size[0],new_size[1], img.shape[2]),img.dtype) # create temporal image

    # Smooth image
    g_vec = GaussianVector((img.shape[0],img.shape[1]), 2)
    fltrd_part =  conv2(img, g_vec,  Pad.REFLECT_ACROSS_EDGE )
    fltrd_img =  conv2(fltrd_part, g_vec.transpose(),  Pad.REFLECT_ACROSS_EDGE )

    fltrd_img.astype(img.dtype)  # convert to np.uint image

    # Downsample image
    for c in range(img.shape[2]): # channel
        down_img[:,:,c] = nn_interpolate(fltrd_img[:,:,c],new_size)

    return down_img



# Gaussian Vector
# MN - image size  M rows, N columns
# perc - image percentage wrt dir 0-100
# dir - direction to get the percentage
def GaussianVector(MN, perc, dir = 'width'):
    # arguments checks
    if dir != 'height' or dir != 'width':
        dir = 'width' # defualt direction

    if perc < 0:
        perc = 0
    elif perc > 100:
        perc = 100

    if len(MN)<2:
        raise TypeError("Argument 'mn' in GaussianVector must have 2 elements")
        exit()
    M, N = MN # separate values

    # determine kernel size
    if dir == 'width':
        ksize = int(N*perc/100)
    else: # dir = height
        ksize = int(M*perc/100)

    # ensure odd ksize
    if not ksize%2: # is ksize even?
        ksize += 1 # make it odd

    # smallest possible kernel
    if ksize < 3:
        ksize = 3

    # determine sigma value
    sigma = 2*ksize/6 # heristics for sigma value

    # smallest possible sigma
    if sigma < 1:
        sigma = 1
    print('   sigma: ' + str(sigma) + ' ksize: ' + str(ksize))

    return cv.getGaussianKernel(ksize,sigma)


# Frequency Response
# Images of the Magnitude and Phase response ready to show with cv.imshow
# f - image (1 cahnnel only)
# normalized - image is normalized to 1 previous DFT
# centered - magnitude spectrum centered? For display only
def freqz(f, normalized=True, centered=True):
    if normalized:
        img_scld = scaleImageChannel(f,1,np.float32) # scale image to range [0,1]
    else:
        img_scld = np.copy(f)
    F = DFT2(img_scld) # compute DFT2
    if centered:
        F_shift = np.fft.fftshift(F)
    else:
        F_shift = np.copy(F)
    log_T = 1+np.abs(F_shift) # apply log tranformation just for visualization
    magnitude_spectrum = np.log(log_T)
    magnitude_spectrum = scaleImage(magnitude_spectrum,modo='custom',K=255)
    phase_angle = np.angle(F_shift)
    phase_angle = scaleImage(phase_angle,modo='custom',K=255)
    return [magnitude_spectrum, phase_angle]




# pyramidBlending Function
def pyramidBlending():
    # define number of layers
    layers = 3

    print("PYRAMID BLENDING\n")

    # Align GUI
    # alg.launch()
    # Mask GUI
    # can.launch()
    # Give some time to finish saving file
    time.sleep(1)

    # default file names
    source = 'foreground.jpg'
    target = 'background.jpg'
    mask = 'mask.jpg'

    #  Source and Target
    S = cv.imread(source)
    T = cv.imread(target)
    M = cv.imread(mask)

    # Start time counter
    start_time = time.time()

    print('Computing S pyramids...')
    gpS, lpS = ComputePyr(S, layers)
    print('Computing T pyramids...')
    gpT, lpT = ComputePyr(T, layers)
    print('Computing M pyramids...')
    gpM, lpM = ComputePyr(M, layers, pyramid='gaussian')

    # Lapacian pyramid for composite
    lpC = []
    for ls,lt,g  in zip(lpS,lpT,gpM):
        lpC_i = np.empty(g.shape,g.dtype) # create temporal image
        for c in range(g.shape[2]): # channel
            gs = (g[:,:,c]/255).astype(np.double) # normalized mask
            # Iterate through pixels
            for y in range(g.shape[0]): #y coordinate
                for x in range(g.shape[1]): # x coordinate
                    # Laplacian pyramid for composite formula
                    # pdb.set_trace()
                    lpC_i[y][x][c] = gs[y][x]*ls[y][x][c] + (1-gs[y][x])*lt[y][x][c]
                    # ensure range TODO
        lpC.append(lpC_i) # add element to pyramid
        # pdb.set_trace()

    # Recontruct image from Laplacian for composite pyramid
    blend_img = lpC[0] #.astype(np.uint8)
    for i in range(1,layers):
        print("Blending layer " + str(i) + ' ...')
        scale = i**2
        blend_img = upSample(blend_img, [lpC[i].shape[0],lpC[i].shape[1]], scale )
        blend_img = cv.add(blend_img, lpC[i]) # add up :)

    # Display and save result
    cv.imshow('Blending',blend_img)
    print('Done!' + str(time.time()-start_time) + 'sec')
    cv.waitKey()
    cv.imwrite('./blending.png',blend_img)
    cv.destroyAllWindows()



# main function
def main():
    # Print module information
    print('''\n
    ____________________________________________________________________________
                                diptools.py
    Functions in this module:
    - conv2(f,w,pad)
    - scaleImageChannel(g,K,data_type)
    - padding(img, pad, mn=(1,1))
    - DFT2(f)
    - IDFT2(F)
    - ComputePyr(input_image, num_layers, pyramid='both')
    - nn_interpolate(A, new_size)
    - upSample(img, new_size, scale)
    - downSample(img, new_size, scale)
    - GaussianVector(MN, perc, dir = 'width')
    - pyramidBlending()

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
        LAPLACIAN_1
        LAPLACIAN_2
        ''')

# Call main function
if __name__ == "__main__":
    main()
