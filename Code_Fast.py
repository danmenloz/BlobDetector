#!/usr/bin/env python
# coding: utf-8

# In[209]:


# import numpy as np
import cv2
# import os
# import math
# import sys
import time
import diptools as dip


# In[384]:


img = cv2.imread('./Images/butterfly.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[385]:

start = time.process_time()
# Creating DOGs
# k = math.sqrt(2) # from paper
k = dip.SQRT2 # from paper
# s0 = math.sqrt(2)/2 # from paper
s0 = dip.SQRT2/2 # from paper
num_DoG = 6
# list_DoG = dip.DoG(img_gray, k, s0, 6)
img_scale = dip.DoG(img_gray, k, s0, 6)
# img_f = np.array(img_gray/255).astype(float)
# jj = []
# # Gaussian blur
# for i in range(0,num_DoG+1):
#     s = s0*(k**i)
#     w = int(2*np.ceil(3*s)+1)
#     jj.append(w)
#     # list_blur.append(cv2.GaussianBlur(img_f,(w,w),s,cv2.BORDER_DEFAULT))
#     kernel = dip.GaussianKernel(w,s)
#     filtered = dip.fftconv2(img_f,kernel)
#     list_blur.append(filtered)
# # Difference of Gaussian
# list_DoG = []
# for i in range(0,num_DoG):
#     imgf_up = list_blur[i+1]
#     imgf_down = list_blur[i]
#     list_DoG.append(abs(imgf_up-imgf_down))


# In[386]:


# # Function to force pixels evaluated are inside image region
# def point_val(p,len):
#     if p-1<0:
#         pi = 0
#         pf = p+1
#     elif p+1>=len:
#         pi = p-1
#         pf = len-1
#     else:
#         pi=p-1
#         pf=p+1
#     return(pi,pf)


# In[387]:


# def dip.mov_arr(arr,pos):
#     if pos == "UP":
#         arr_out = np.roll(arr,1,axis=1) # UP
#         arr_out[:,0,:] = 0
#     elif pos == "DOWN":
#         arr_out = np.roll(arr,-1,axis=1) # DOWN
#         arr_out[:,arr.shape[1]-1,:] = 0
#     elif pos == "LEFT":
#         arr_out = np.roll(arr,1,axis=2) # LEFT
#         arr_out[:,:,0] = 0
#     elif pos == "RIGHT":
#         arr_out = np.roll(arr,-1,axis=2) # RIGHT
#         arr_out[:,:,arr.shape[2]-1] = 0
#     else:
#         arr_out = arr # NO MOVE
#     return arr_out


# In[388]:


# def laplacian_blob_fast(img_scale,S_limit):
#     img_scale_1 = dip.mov_arr(img_scale,"UP")
#     img_scale_0 = dip.mov_arr(img_scale_1,"LEFT")
#     img_scale_2 = dip.mov_arr(img_scale_1,"RIGHT")
#     img_scale_7 = dip.mov_arr(img_scale,"DOWN")
#     img_scale_6 = dip.mov_arr(img_scale_7,"LEFT")
#     img_scale_8 = dip.mov_arr(img_scale_7,"RIGHT")
#     img_scale_3 = dip.mov_arr(img_scale,"LEFT")
#     img_scale_5 = dip.mov_arr(img_scale,"RIGHT")
#
#     img_scale_T = np.concatenate((img_scale,img_scale_0,img_scale_1,img_scale_2,img_scale_3,
#                          img_scale_5,img_scale_6,img_scale_7,img_scale_8)) # Region for NMS evaluation
#
#     img_scale_s = np.argmax(img_scale_T,axis=0) # Scale position of maximum value
#     int_kpoint = np.logical_and((1<=img_scale_s),(img_scale_s<(img_scale.shape[0]-1))) # Verifies if max value inside Region
#     scale_max = np.max(img_scale_T,axis=0) # Scale maximum values matrix
#
#     ms = img_scale_s*int_kpoint # Scale Value
#     val_kpoint = scale_max*int_kpoint # Scale maximum values matrix inside region, otherwise scale=0
#     xy_kpoint = int_kpoint*(val_kpoint>=S_limit) # Value limitation for keypoint values
#     r_kpoint = (s0*(math.sqrt(2)**ms))*math.sqrt(2) # Circle Radius
#     c_kpoint = np.where(xy_kpoint==1) # Saving pixel position
#     return xy_kpoint,c_kpoint,r_kpoint


# In[393]:


# Finding Integer Key Points
# start = time.process_time()
# img_scale = np.stack(list_DoG[0:num_DoG]) # Converts list in 3d matrix
S_limit = 0.07
int_kpoint,c_kpoint,r_kpoint = dip.lapacianBlob(img_scale,S_limit,s0)
ptime = time.process_time() - start
print("Execution time(sec): " + str(ptime))


# In[394]:


# # Presenting integer keypoints in Image
# intxy_kp = (int_kpoint>0) # integer keypoints positions
# intnum_kp = np.sum(intxy_kp!=0) # Number of all keypoints
# print("Total keypoints: ",intnum_kp)


# In[395]:


# Draw Circles
K = 3 # radius factor
r_limit = 300 # radius limit
# img_out = img.copy()
# c_int = np.array(c_kpoint).astype(int)
# r_int = K*np.array(r_kpoint).astype(int)
# for d2 in range(0,img.shape[1]):
#     for d1 in range(0,img.shape[0]):
#         if (intxy_kp[d1,d2]):
#             r_int[r_int>r_limit] = r_limit
#             cv2.circle(img_out, tuple([d2,d1]), r_int[d1,d2],(0,255,0), 1)

intnum_kp, img_out = dip.drawCircles(img, K, r_limit, int_kpoint, c_kpoint, r_kpoint)

name = "Circles: " + str(intnum_kp)
cv2.imshow(name,img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:
