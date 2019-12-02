import numpy as np
from scipy import misc
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2 as cv
import diptools as dip

# img = misc.face()[:,:,0]
img_file = './Images/einstein.jpg'
img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
# img = np.ones((16,16))

# Kernel must be square!
# kernel = np.ones((5,5)) #/ 9
# kernel = np.array([[1,2,3],[4,5,6],[7,8,9]])
# kernel = np.ones((7,7)) / 49
kernel = dip.GaussianKernel(31,3)

# # Kernel padding
# sz = (img.shape[0] - kernel.shape[0], img.shape[1] - kernel.shape[1])  # total amount of padding
# y1,y2,x1,x2 = ((sz[0]+1)//2, sz[0]//2, (sz[1]+1)//2, sz[1]//2) # padding on each side
# kernel1 = np.pad(kernel, ((y1, y2), (x1, x2)), 'wrap')
# kernel2 = dip.padding(kernel, dip.Pad.WRAP_AROUND, mn=(y1+1,x1+1))
#
# # resize kernel to fit img size
# if y1>y2 :
#     kernel2 = kernel2[0:img.shape[0],:]
# if x1>x2 :
#     kernel2 = kernel2[:,0:img.shape[1]]
#
# import pdb; pdb.set_trace()
#
#
# kernel2 = fftpack.ifftshift(kernel2)
#
# # filtered = np.real(fftpack.ifft2(fftpack.fft2(img) * fftpack.fft2(kernel)))
# filtered = np.real(dip.IDFT2(dip.DFT2(img) * dip.DFT2(kernel2)))

filtered = dip.fftconv2(img,kernel)

print('Max: ' + str(np.max(filtered)))
print('Min: ' + str(np.min(filtered)))

plt.imshow(filtered, vmin=0, vmax=255)
plt.show()
# cv.imshow('filtered', dip.scaleImage(filtered, modo='custom'))
# print('Done!')
# cv.waitKey()
# cv.destroyAllWindows()
