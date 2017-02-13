##Writeup 
[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Distorted"
[image2]: ./camera_cal/calibration_completed1.jpg "Undistorted"
[image3]: ./test_images/test1.jpg "Test"
[image4]: ./test_images/tracked0.jpg "Tracked"
[image5]:  ./test_images/tracked_binary0.jpg "Binary Color"
[image6]:  ./test_images/first_warp0.jpg "Perspective Transform"
[image7]:  ./test_images/first_warp0.jpg "Line Pixels"
[image8]: ./test_images/road_tracked0.jpg "Tracking Lanes"
[image9]: ./test_images/road_tracked1.jpg "Tracking Lanes"
[image10]: ./test_images/warped0.jpg "Windows"
[image11]: ./test_images/warped1.jpg "Windows"
[image12]:  ./test_images/road_warped0.jpg "Lane Curvature"
[image13]:  ./test_images/road_warped1.jpg "Lane Curvature"


[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

###Writeup / README

###Camera Calibration

The 'camera_cal' folder contais a camera_cal.py file to 
distort the images and learn the distortion coefficients to employ with OpenCV's 'cv2.undistort()' function.

Firstly, we create two arrays in which to store the coordinated of the chessboard corners. We will apend 'objpoints' each time we detect all corners in test images of chessboards, and add the (x, y) pixel position of the corners to 'imgpoints'.

Here we can see the difference between the images. The first is distorted, with a slight bends around the edges of the image. The second image shows the undistorted picture of the board. 

Distorted
![alt text][image1]

Undistorted
![alt text][image2]


###Pipeline (single images)

We can now apply the distortion to one of the test images from the road lane we are interested in for this project. Here's the original image:
![alt text][image3]

Here's the undistorted image. Notice the view of the white car to the right of the screen:
![alt text][image4]

**Binary color transform

To process the test images into thresholded binary images, I use a combination of color and gradient threshold methods. These techniques take us beyond a simple edge detection and allow us to find which lines are more likely to be lanes based on their length and curvature.
To this end, we use Sobel operators to take the derivative of the image in the x or y direction.
```
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    # Apply threshold
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output
```

Then we apply an overall threshold to the magnitude of the gradient.

```
def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    # Apply threshold
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output
```

We then work up directional thresholds to allow the program to docus on detecting lane lines rather than all lines (other cars, bridges, traffic features etc). 

```
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    with np.errstate(divide='ignore', invalid='ignore'):
        absgraddir = np.absolute(np.arctan(sobely/sobelx))
        binary_output =  np.zeros_like(absgraddir)
        # Apply threshold
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output
```

In the above code, we calculate the direction as the arctangent of the y-gradient divided by the x-gradient. 

Here is the image from test1 converted to the binary threshold using the above code segments:

![alt text][image5]

We then use OpenCV methods to convert the RGB images to HLS, which affords greater distinction between features in an image due to the lightness and saturation of pixels.

```
def color_threshold(image, sthresh=(0,255), vthresh=(0,255)):
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])  ] = 1

	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	v_channel = hsv[:,:,2]
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])  ] = 1

	output = np.zeros_like(s_channel)
	output[(s_binary == 1) & (v_binary == 1)] = 1
	return output
```

A mask then focuses the image in the center of the picture, disgarding some of the dashboard and the sky.

```
def window_mask(width, height, img_ref, center,level):
		output = np.zeros_like(img_ref)
		output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
		return output
```

The next major stage for image processing is to warp the image. This means we take the view of the road ahead, and make it stand upright, as if we were looking at the road from a bird's eye view. This of course should make our lane detection more accurate, since the lines run from top to bottom, rather than into the distance.

I experimented with various widths and heights for the size of the trapizoid before running with the following:

```
# work on defining perspective transformation area
img_size = (img.shape[1],img.shape[0])
bot_width = .75 # percent of bottom trapizoid height
mid_width = .15 # percent of middle trapizoid height
height_pct = .65 # percent for trapizoid height -- sets extent of range down the road
bottom_trim = .935 # percent from top to bottom to avoid car hood 
src = np.float32([[img.shape[1]*(.5-mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(.5+mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(.5+bot_width/2),img.shape[0]*bottom_trim],[img.shape[1]*(.5-bot_width/2),img.shape[0]*bottom_trim]])
offset = img_size[0]*.25
dst = np.float32([[offset, 0], [img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]], [offset ,img_size[1]]])

# perform the transform
M = cv2.getPerspectiveTransform(src,dst)
Minv = cv2.getPerspectiveTransform(dst,src)
warped = cv2.warpPerspective(preprocessImage,M,img_size,flags=cv2.INTER_LINEAR)

result = warped

write_name = './test_images/first_warp'+str(idx)+'.jpg'
cv2.imwrite(write_name, result)

```

The warp function results in an image like this:

![alt text][image7]

I then fit polynomials to decide which of the lines marked in the processed image are part of the left and right lanes. Overlaying windows onto the left and right lane pixels helps us to track the lines from the bottom to the top of the image.

```
# Draw windows for each level
	for level in range(0, len(window_centroids)):
		# window_mask function draws window areas
		l_mask = window_mask(window_width,window_height, warped, window_centroids[level][0],level)
		r_mask = window_mask(window_width,window_height, warped, window_centroids[level][1],level)
		# Add center value to right, left lane points lists
		leftx.append(window_centroids[level][0])
		rightx.append(window_centroids[level][1])
		# Add graphic points from window mask to total pixels found
		l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
		r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

	# Fit curves to images
	# fit the lane boundaries to the left,right center positions found
	yvals = range(0,warped.shape[0])

	res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)

	left_fit = np.polyfit(res_yvals, leftx, 2)
	left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
	left_fitx = np.array(left_fitx,np.int32)
	
	right_fit = np.polyfit(res_yvals, rightx, 2)
	right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
	right_fitx = np.array(right_fitx,np.int32)

	# used to format everything so its ready for cv2 draw functions
	left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
	right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
	middle_marker = np.array(list(zip(np.concatenate((right_fitx-window_width/2, right_fitx[::-1]+window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

	# draw lane lines, middle curve, road background on two different blank overlays
	road = np.zeros_like(img)
	road_bkg = np.zeros_like(img)
	cv2.fillPoly(road,[left_lane],color=[255, 0, 0])
	cv2.fillPoly(road,[right_lane],color=[0, 0, 255])
	cv2.fillPoly(road_bkg,[left_lane],color=[255, 255, 255])
	cv2.fillPoly(road_bkg,[right_lane],color=[255, 255, 255])
	
	road_warped = cv2.warpPerspective(road,Minv,img_size,flags=cv2.INTER_LINEAR)
	road_warped_bkg = cv2.warpPerspective(road_bkg,Minv,img_size,flags=cv2.INTER_LINEAR)

	base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
	result = cv2.addWeighted(base, 1.0, road_warped, 1.0, 0.0)

	ym_per_pix = curve_centers.ym_per_pix # meters/pixel in y dimension
	xm_per_pix = curve_centers.xm_per_pix # meters/pixel in x dimension

	curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix, 2)
	curverad = ((1 + (2*curve_fit_cr[0]*yvals[1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) /np.absolute(2*curve_fit_cr[0])
```
Coloring left lane blue and right lane red 
![alt text][image8]

![alt text][image9]

Adding windows to track lane line
![alt text][image10]

![alt text][image11]

Overlaying curvature onto image
![alt text][image12]

![alt text][image13]


###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output1_tracked.mp4)

---

###Discussion
 
While the pipeline works well on the video shown, the lanes are generally progressing forwards and to the left. Further work and flexibility will be necessary for the algorithm to accurately pick up quickly changing right and left swerves in lines on the road. 

This was a challenging project. I found working out some of the mathematics involved in Python took a lot of research, and making sure the various image processing techniques could do their work, pass the image on to the next processing function, took a fair amount of effort with plenty of bug fixing. However, this was a fun and very engaging project, preceeded by some great lessons in computer vision. Thank you Udacity. Avanti!
