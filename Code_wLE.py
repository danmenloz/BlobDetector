#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import cv2
import os
import math
import sys


# In[74]:


img = cv2.imread('butterfly.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[99]:


# Creating DOGs
list_blur = []
k = math.sqrt(2) # from paper
s0 = math.sqrt(2)/2 # from paper
img_f = np.array(img_gray/255).astype(float)
num_DoG = 8
w = [13,17,21,27,35,43,55,69]
jj = []
# Gaussian blur
for i in range(0,num_DoG+1):
    s = s0*(k**i)
    w = int(2*np.ceil(3*s)+1)
    jj.append(w)
    list_blur.append(cv2.GaussianBlur(img_f,(w,w),s,cv2.BORDER_DEFAULT))
# Difference of Gaussian
list_DoG = []
for i in range(0,num_DoG):
    imgf_up = list_blur[i+1]
    imgf_down = list_blur[i]
    list_DoG.append(abs(imgf_up-imgf_down))


# In[86]:


# Finding Integer Key Points
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


# In[87]:


# Lapalcian Blob function
# Main LOOP
def laplacian_blob(img_scale,D_limit,R_limit):
    int_kpoint = np.zeros((img_scale.shape[1],img_scale.shape[2]))
    lowc_kpoint = np.zeros((img_scale.shape[1],img_scale.shape[2]))
    edge_kpoint = np.zeros((img_scale.shape[1],img_scale.shape[2]))
    r_kpoint = np.zeros((img_scale.shape[1],img_scale.shape[2]))
    c_kpoint = np.zeros((img_scale.shape[1],img_scale.shape[2],2))
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
                # All keypoints
                int_kpoint[d1,d2] = 1
                # Gradient Calculation (Central Gradient)
                grad_x = (img_scale[ms,xf,d2]-img_scale[ms,xi,d2])/2
                grad_y = (img_scale[ms,d1,yf]-img_scale[ms,d1,yi])/2
                grad_s = (img_scale[ms+1,d1,d2]-img_scale[ms-1,d1,d2])/2
                # Hessian Calculation
                h11 = img_scale[ms,xf,d2]-2*img_scale[ms,d1,d2]+img_scale[ms,xi,d2]
                h12 = (img_scale[ms,xf,yf]-img_scale[ms,xi,yf]-img_scale[ms,xf,yi]+img_scale[ms,xi,yi])/4
                h13 = (img_scale[ms+1,xf,d2]-img_scale[ms+1,xi,d2]-img_scale[ms-1,xf,d2]+img_scale[ms-1,xi,d2])/4
                h21 = h12
                h22 = img_scale[ms,d1,yf]-2*img_scale[ms,d1,d2]+img_scale[ms,d1,yi]
                h23 = (img_scale[ms+1,d1,yf]-img_scale[ms+1,d1,yi]-img_scale[ms-1,d1,yf]+img_scale[ms-1,d1,yi])/4
                h31 = h13
                h32 = h23
                h33 = img_scale[ms+1,d1,d2]-2*img_scale[ms,d1,d2]+img_scale[ms-1,d1,d2]
                # Offset point Calculation (Interpolation)
                H = np.array([[h11,h12,h13],[h21,h22,h23],[h31,h32,h33]]) # Hessian Matrix
                if (abs(np.linalg.det(H))>0.0001): # Verify if H is Singular
                    p_off = -np.matmul(np.linalg.inv(H),[[grad_x],[grad_y],[grad_s]])
                else:
                    p_off = [0,0,0] # (VALIDAR!!!!)
                c_kpoint[d1,d2,:] = [d1+p_off[0],d2+p_off[1]] # Subpixel maxima
                r_kpoint[d1,d2] = (s0*(math.sqrt(2)**ms)+p_off[2])*math.sqrt(2) # Circle Radius        
                # Verifying Low contrast
                p_lowc = img_scale[ms,d1,d2]+0.5*np.matmul([grad_x,grad_y,grad_s],p_off)
                if (p_lowc>=D_limit):
                    lowc_kpoint[d1,d2] = 1
                # Verifying Edges
                det_M = h11*h22-h12*h21 # Determinant of 2x2 Hessian Matrix
                tra_M = h11+h22 # Trace of 2x2 Hessian Matrix
                ratio = tra_M**2/det_M # Ratio calculation
                if (det_M>=0 and ratio<=R_limit):
                    edge_kpoint[d1,d2] = 1
    return int_kpoint,lowc_kpoint,edge_kpoint,c_kpoint,r_kpoint


# In[107]:


# Se asume que en los bordes solo existen 17 vecinos y en las esquinas 5 vecinos
# Que sucede si hay dos neighbours maximos? Uno en el centro y otro en el extremo
img_scale = np.stack(list_DoG[0:num_DoG]) # Converts list in 3d matrix. 
D_limit = 0.03 # From paper
R_limit = 4.5 # (Set by observation - low value -> less blobs)
int_kpoint,lowc_kpoint,edge_kpoint,c_kpoint,r_kpoint = laplacian_blob(img_scale,D_limit,R_limit)


# In[108]:


# Presenting integer keypoints in Image
intxy_kp = (int_kpoint>0) # integer keypoints positions
lowcxy_kp = (lowc_kpoint>0) # low contrast keypoints positions
edgexy_kp = (edge_kpoint>0) # no edge keypoints positions
edgelowcxy_kp = np.logical_and(edgexy_kp,lowcxy_kp) # no edge and low contrast keypoints positions

intnum_kp = np.sum(intxy_kp!=0) # Number of all keypoints
lowcnum_kp = np.sum(lowcxy_kp!=0) # Number of low contrast keypoints
edgenum_kp = np.sum(edgexy_kp!=0) # Number of no edge keypoints
edgelowcnum_kp = np.sum(edgelowcxy_kp!=0) # Number of no edge and low contrast keypoints

print("Total keypoints: ",intnum_kp)
print("Total keypoints (Discarding low contrast): ",lowcnum_kp)
print("Total keypoints (Discarding edges): ",edgenum_kp)
print("Total keypoints (Discarding low contrast and edges): ",edgelowcnum_kp)


# In[109]:


# Draw Circles
K = 3 # radius factor
r_limit = 30 # radius limitation
img_out = img.copy()
c_int = np.array(c_kpoint).astype(int)
r_int = K*np.array(r_kpoint).astype(np.uint8)
for d2 in range(0,img.shape[1]):
    for d1 in range(0,img.shape[0]):
        if (edgelowcxy_kp[d1,d2]):
            x_pos = c_int[d1,d2,1]
            y_pos = c_int[d1,d2,0]
            r_int[r_int>r_limit] = r_limit
            cv2.circle(img_out, tuple([x_pos,y_pos]), r_int[d1,d2],(0,255,0), 1)

name = "Circles: " + str(edgelowcnum_kp)
cv2.imshow(name,img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




