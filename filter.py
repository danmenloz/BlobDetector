import diptools as dip
import numpy as np
import cv2 as cv
import pdb
import time

nrows = 150
ncols = 150

Huv = np.empty((nrows,ncols),np.double)

for v in range(nrows):
    for u in range(ncols):
        Huv[v][u]=np.exp(-(u**2/150+v**2/150)) +  1-np.exp(-((u-50)**2/150+(v-50)**2/150))

cv.imshow('Huv Filter', dip.scaleImage(Huv,modo='custom',K=255)) #np.fft.fftshift(Huv)
print('Done!')
cv.waitKey()
cv.imwrite('Spectrum.png', dip.scaleImage(Huv,modo='custom',K=255))
cv.destroyAllWindows()
