 # Project 04 Digital Imaging Processing
# Authors:
#   * Daniel Mendez         (dmendez@ncsu.edu)
#   * Alhiet Orbegoso       (aoorbego@ncsu.edu)
#   * Jacqueline Almache     (jalmach@ncsu.edu)
# Version: python 3.7.4
# Description: This file contains the core functions developed in all previous
#   projects.

import numpy as np
import cv2 as cv
from enum import Enum

# Constants
SQRT2 = 2**(1/2)

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



# Fast Fourier Transform 2D
# f - Grayscale input image
def dft2(f):
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
def idft2(f):
    f = np.array(f)
    # Calculating input matrix size
    size_x = f.shape[0]
    size_y = f.shape[1]
    # Conjugate input matrix
    conj_f = np.conj(f)
    # Calculating inverse fft
    idft_2 = 1/(size_x*size_y)*dft2(conj_f)
    # return idft_2.real
    return idft_2



# Conv2 using fft implementation
# conv2 implemented in frequency domain
# image - grayscale image
# kernel - gaussian symmetric kernel or other symmetric kernel
def fftconv2(img, kernel):
    # Kernel padding
    sz = (img.shape[0] - kernel.shape[0], img.shape[1] - kernel.shape[1])  # total amount of padding
    y1,y2,x1,x2 = ((sz[0]+1)//2, sz[0]//2, (sz[1]+1)//2, sz[1]//2) # padding on each side
    kernel = padding(kernel, Pad.CLIP_ZERO, mn=(y1+1,x1+1)) # pad the kernel

    # resize kernel to fit img size
    if y1>y2 :
        kernel = kernel[0:img.shape[0],:]
    if x1>x2 :
        kernel = kernel[:,0:img.shape[1]]

    kernel = np.fft.fftshift(kernel) # shift center to origin (top left corner)

    filtered = np.real(idft2(dft2(img) * dft2(kernel))) # conv2 in freq domain

    return filtered # convolved image



# Nearest Neighbor Interpolation
# A - np.array image
# new_size - list [height,width]
def nnInterpolate(A, new_size):
    old_size = A.shape
    row_ratio, col_ratio = np.array(new_size)/np.array(old_size)

    # row wise interpolation
    row_idx = (np.ceil(range(1, 1 + int(old_size[0]*row_ratio))/row_ratio) - 1).astype(int)

    # column wise interpolation
    col_idx = (np.ceil(range(1, 1 + int(old_size[1]*col_ratio))/col_ratio) - 1).astype(int)

    final_matrix = A[:, col_idx][row_idx, :] #modified

    return final_matrix



# Upsample image
# img - np image arrays
# new_size - size [height, width] of the output image
# scale - downscale or upscale as a power of 2: 2,4,6...
def upSample(img, new_size, scale):
    up_img = np.empty((new_size[0],new_size[1], img.shape[2]),img.dtype) # create temporal image

    # Upsample image
    for c in range(img.shape[2]): # channel
        up_img[:,:,c] = nnInterpolate(img[:,:,c],new_size)

    # Smooth image
    # g_vec = gaussianVector2(scale)
    g_vec = gaussianVector((up_img.shape[0],up_img.shape[1]), 2)
    fltrd_part =  conv2(up_img, g_vec,  Pad.REFLECT_ACROSS_EDGE )
    fltrd_img =  conv2(fltrd_part, g_vec.transpose(),  Pad.REFLECT_ACROSS_EDGE )

    return fltrd_img.astype(img.dtype)  # convert to np.uint image



# Downsample image
# img - np image array
# new_size - size [height, width] of the output image
# scale - downscale or upscale as a power of 2: 2,4,6...
def downSample(img, new_size, scale):
    down_img = np.empty((new_size[0],new_size[1], img.shape[2]),img.dtype) # create temporal image

    # Smooth image
    g_vec = gaussianVector((img.shape[0],img.shape[1]), 2)
    fltrd_part =  conv2(img, g_vec,  Pad.REFLECT_ACROSS_EDGE )
    fltrd_img =  conv2(fltrd_part, g_vec.transpose(),  Pad.REFLECT_ACROSS_EDGE )

    fltrd_img.astype(img.dtype)  # convert to np.uint image TODO: fix line

    # Downsample image
    for c in range(img.shape[2]): # channel
        down_img[:,:,c] = nnInterpolate(fltrd_img[:,:,c],new_size)

    return down_img



# Gaussian Vector for filtering
# MN - image size  M rows, N columns
# perc - image percentage wrt dir 0-100
# dir - direction to get the percentage
def gaussianVector(MN, perc, dir = 'width'):
    # arguments checks
    if dir != 'height' or dir != 'width':
        dir = 'width' # defualt direction

    if perc < 0:
        perc = 0
    elif perc > 100:
        perc = 100

    if len(MN)<2:
        raise TypeError("Argument 'mn' in gaussianVector must have 2 elements")
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



# Gaussian Kernel for filtering
# ksize - size of the square kernel
# sigma - spread kernel value
def gaussianKernel(ksize,sigma):
    # if not ksize%2: # is ksize even?
    #     raise TypeError("kernel size must be odd")
    #     exit()
    g = cv.getGaussianKernel(ksize,sigma)
    G = np.multiply(g,np.transpose(g))
    return(G)



# Difference of Gaussians
# img - initial image to be blurred (grayscale)
# k - subdivision between blurred images
# sigma - initial standard deviation
# scales - num of DoG returned
def DoG(img, k, sigma, scales):
    img_n = np.array(img/255).astype(float) # normalize image
    # Image smoothing
    list_blur = []
    for i in range(0,scales+1):
        sigma_n = sigma*(k**i) # sigma value for this layer
        ksize = int(2*np.ceil(3*sigma_n)+1) # change kernel size
        kernel = gaussianKernel(ksize,sigma_n) # generate Gaussian kernel
        blur = fftconv2(img_n,kernel) # blurr image
        list_blur.append(blur) # add image to list
    # Difference of Gaussian
    list_DoG = []
    for i in range(0,scales):
        imgf_up = list_blur[i+1] # upper blurred image
        imgf_down = list_blur[i] # lower blurred image
        list_DoG.append(abs(imgf_up-imgf_down)) # compute DoG
    img_scale = np.stack(list_DoG[0:scales]) # Converts list in 3d matrix
    return img_scale



# Move Array
# arr - input Array
# pos - Direction
def moveArray(arr, pos):
    if pos == "UP":
        arr_out = np.roll(arr,1,axis=1) # UP
        arr_out[:,0,:] = 0
    elif pos == "DOWN":
        arr_out = np.roll(arr,-1,axis=1) # DOWN
        arr_out[:,arr.shape[1]-1,:] = 0
    elif pos == "LEFT":
        arr_out = np.roll(arr,1,axis=2) # LEFT
        arr_out[:,:,0] = 0
    elif pos == "RIGHT":
        arr_out = np.roll(arr,-1,axis=2) # RIGHT
        arr_out[:,:,arr.shape[2]-1] = 0
    else:
        arr_out = arr # NO MOVE
    return arr_out



# Compute Lapacian Blobs using non-max supression
# img_scale - DoG sapace
# S_limit - Value limit for keypoint values
# sigma - standard deviation value
def lapacianBlob(img_scale, S_limit, sigma):
    img_scale_1 = moveArray(img_scale,"UP")
    img_scale_0 = moveArray(img_scale_1,"LEFT")
    img_scale_2 = moveArray(img_scale_1,"RIGHT")
    img_scale_7 = moveArray(img_scale,"DOWN")
    img_scale_6 = moveArray(img_scale_7,"LEFT")
    img_scale_8 = moveArray(img_scale_7,"RIGHT")
    img_scale_3 = moveArray(img_scale,"LEFT")
    img_scale_5 = moveArray(img_scale,"RIGHT")

    img_scale_T = np.concatenate((img_scale,img_scale_0,img_scale_1,img_scale_2,img_scale_3,
                         img_scale_5,img_scale_6,img_scale_7,img_scale_8)) # Region for NMS evaluation

    img_scale_s = np.argmax(img_scale_T,axis=0) # Scale position of maximum value
    int_kpoint = np.logical_and((1<=img_scale_s),(img_scale_s<(img_scale.shape[0]-1))) # Verifies if max value inside Region
    scale_max = np.max(img_scale_T,axis=0) # Scale maximum values matrix

    ms = img_scale_s*int_kpoint # Scale Value
    val_kpoint = scale_max*int_kpoint # Scale maximum values matrix inside region, otherwise scale=0
    xy_kpoint = int_kpoint*(val_kpoint>=S_limit) # Value limitation for keypoint values
    r_kpoint = (sigma*(SQRT2**ms))*SQRT2 # Circle Radius
    c_kpoint = np.where(xy_kpoint==1) # Saving pixel position
    return xy_kpoint,c_kpoint,r_kpoint


# Draw Circles on an image
# img - image to display
# K - radius factor
# r_limit - radius limit
# xy_kpoint - keypoints position
# c_kpoint - keypoints center
# r_kpoint - kepoints radius
def drawCircles(img, K, r_limit, xy_kpoint, c_kpoint, r_kpoint):
    intxy_kp = (xy_kpoint>0) # integer keypoints positions
    intnum_kp = np.sum(intxy_kp!=0) # Number of all keypoints
    img_out = img.copy() # copy input image
    c_int = np.array(c_kpoint).astype(int) # circle center
    r_int = K*np.array(r_kpoint).astype(int) # circle radius
    # draw circle in image
    for d2 in range(0,img.shape[1]):
        for d1 in range(0,img.shape[0]):
            if (intxy_kp[d1,d2]):
                r_int[r_int>r_limit] = r_limit
                cv.circle(img_out, tuple([d2,d1]), r_int[d1,d2],(0,255,0), 1)
    return intnum_kp, img_out



# Scale Image for display
# img - input image
# modo - 'auto' uses the max(img) as limit, 'custom' uses K parameter as limit
# K - maximum intensity when usidin modo 'custom'
# crop - crop negative values to zero
def scaleImage(img, modo = 'auto', K = 255, crop=False):
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



# Scaled image with spanned intensities in the range [0,K]
# g         - 1D image
# K         - maximum value
# data_type - np data type for the result
def scaleImageChannel(g,K,data_type):
    # see page 91 of DIP book for reference
    min_g = g.min()
    g_aux = g.astype(np.float32) # copy of g as float to avoid overflow

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
# Note: img is meant to be bigger than the kernel size mn
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
                import pdb; pdb.set_trace()
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



# Frequency Response
# Images of the Magnitude and Phase response ready to show with cv.imshow
# f - image (1 cahnnel only)
# normalized - image is normalized to 1 previous DFT
# centered - magnitude spectrum centered? For display only
def freqz(f, normalized=True, centered=True):
    if normalized:
        img_scld = scaleImageChannel(f,1,np.float64) # scale image to range [0,1]
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


# main function
def main():
    # Print module information
    print('''\n
    ____________________________________________________________________________
                                diptools.py
    Functions in this module:
    - conv2(f,w,pad)
    - dft2(f)
    - idft2(f)
    - fftconv2(img, kernel)
    - nnInterpolate(A, new_size)
    - upSample(img, new_size, scale)
    - downSample(img, new_size, scale)
    - gaussianVector(MN, perc, dir = 'width')
    - gaussianKernel(ksize,sigma)
    - DoG(img, k, sigma, scales)
    - moveArray(arr, pos)
    - lapacianBlob(img_scale, S_limit, sigma)
    - drawCircles(img, K, r_limit, xy_kpoint, c_kpoint, r_kpoint)
    - scaleImage(img, modo = 'auto', K = 255, crop=False)
    - scaleImageChannel(g,K,data_type)
    - padding(img, pad, mn=(1,1))
    - freqz(f, normalized=True, centered=True)



    Enumerations  and Constants in this module:
    - SQRT2 = 2**(1/2)
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
