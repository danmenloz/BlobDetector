import cv2
import time
import distools as dip

# Read image
img = cv2.imread('./Images/butterfly.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

start = time.process_time() # Start time counter

# Creating DOG space
k = dip.SQRT2 # subdivision between blurred images
s0 = dip.SQRT2/2 # initial sigma value
num_DoG = 6 # number of layers
img_scale = dip.DoG(img_gray, k, s0, num_DoG)

# Finding Key Points
S_limit = 0.07 # Value limit for keypoint values
int_kpoint,c_kpoint,r_kpoint = dip.lapacianBlob(img_scale,S_limit,s0)

# Draw Circles
K = 3 # radius factor
r_limit = 300 # radius limit
intnum_kp, img_out = dip.drawCircles(img, K, r_limit, int_kpoint, c_kpoint, r_kpoint)

ptime = time.process_time() - start # End time counter
print("Execution time(sec): " + str(ptime))

# Show result
name = "Circles: " + str(intnum_kp)
cv2.imshow(name,img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
