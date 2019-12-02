#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import cv2
import os
import math
import sys
import diptools as dip
import time


# In[219]:


img = cv2.imread('./Images/butterfly.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[ ]:


# Creating DOGs
list_blur = []
k = math.sqrt(2) # from paper
s0 = math.sqrt(2)/2 # from paper
img_f = np.array(img_gray/255).astype(float) # normalize image
num_DoG = 6
# w = [13,17,21,27,35,43,55,69]
# jj = []
# Gaussian blur
for i in range(0,num_DoG+1):
    s = s0*(k**i) # sigma value for this layer
    w = int(2*np.ceil(3*s)+1) # change kernel size
    # print('w: ' + str(w))
    # jj.append(w)
    # list_blur.append(cv2.GaussianBlur(img_f,(w,w),s,cv2.BORDER_DEFAULT))
    kernel = dip.GaussianKernel(w,s)
    filtered = dip.fftconv2(img_f,kernel)
    list_blur.append(filtered)
    # import pdb; pdb.set_trace()

# Difference of Gaussian
list_DoG = []
for i in range(0,num_DoG):
    imgf_up = list_blur[i+1]
    imgf_down = list_blur[i]
    list_DoG.append(abs(imgf_up-imgf_down))
# np.max(list_DoG[0])


# In[239]:


# Function to force pixels evaluated are inside image region
def point_val(p,len):
    if p-1<0:
        pi = 0
        pf = p+1
    elif p+1>=len:
        pi = p-1
        pf = len-1
    else:
        pi=p-1
        pf=p+1
    return(pi,pf)


# In[240]:


# Laplacian Blob main function
def laplacian_blob(img_scale,S_limit):
    int_kpoint = np.zeros((img_scale.shape[1],img_scale.shape[2]))
    val_kpoint = np.zeros((img_scale.shape[1],img_scale.shape[2]))
    r_kpoint = np.zeros((img_scale.shape[1],img_scale.shape[2]))
    c_kpoint = np.zeros((img_scale.shape[1],img_scale.shape[2],2))
    # Main LOOP
    for d2 in range(0,img_scale.shape[2]):
        for d1 in range(0,img_scale.shape[1]):
            (xi,xf)=point_val(d1,img_scale.shape[1]) # Forcing X pixels inside img region
            (yi,yf)=point_val(d2,img_scale.shape[2]) # Forcing Y pixels inside img region
            max_neighbour = np.max(img_scale[:,xi:xf+1,yi:yf+1]) # Maximum value of all neighbours
            min_neighbour = np.min(img_scale[:,xi:xf+1,yi:yf+1]) # Minimum value of all neighbours
            max_isIn = sum(img_scale[1:num_DoG-1,d1,d2]>=max_neighbour) # Verifying if maximum is inside neighbour region
            min_isIn = sum(img_scale[1:num_DoG-1,d1,d2]<=min_neighbour) # Verifying if minimum is inside neighbour region
            if (max_isIn>0 or min_isIn>0):
                # Get maximum scale location
                ms = np.argmax(abs(img_scale[1:num_DoG-1,d1,d2]))+1
                val_kpoint[d1,d2] = img_scale[ms-1,d1,d2]
                if val_kpoint[d1,d2] > S_limit:
                    # All keypoints
                    int_kpoint[d1,d2] = 1
                    c_kpoint[d1,d2,:] = [d1,d2] # Saving pixel position
                    r_kpoint[d1,d2] = (s0*(math.sqrt(2)**ms))*math.sqrt(2) # Circle Radius
    return int_kpoint,c_kpoint,r_kpoint


# In[241]:


# Finding Integer Key Points
start = time.process_time()
img_scale = np.stack(list_DoG[0:num_DoG]) # Converts list in 3d matrix. index=0 in list, index=0 in matrix, index 1 = Xdim, index2 = Ydim
S_limit = 0.07
int_kpoint,c_kpoint,r_kpoint = laplacian_blob(img_scale,S_limit)
ptime = time.process_time() - start
print("Execution time(sec): " + str(ptime))


# In[242]:


# Presenting integer keypoints in Image
intxy_kp = (int_kpoint>0) # integer keypoints positions
intnum_kp = np.sum(intxy_kp!=0) # Number of all keypoints
print("Total keypoints: ",intnum_kp)


# In[243]:


# Draw Circles
K = 3 # radius factor
r_limit = 30 # radius limitation
img_out = img.copy()
c_int = np.array(c_kpoint).astype(int)
r_int = K*np.array(r_kpoint).astype(np.uint8)
for d2 in range(0,img.shape[1]):
    for d1 in range(0,img.shape[0]):
        if (intxy_kp[d1,d2]):
            x_pos = c_int[d1,d2,1]
            y_pos = c_int[d1,d2,0]
            r_int[r_int>r_limit] = r_limit
            cv2.circle(img_out, tuple([x_pos,y_pos]), r_int[d1,d2],(0,255,0), 1)
name = "Circles: " + str(intnum_kp)
cv2.imshow(name,img_out)
#cv2.imshow('BLUR',(list_DoG[0]*255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
