
'''
With great thanks to Udacity Content Developer Aaron Brown for his excellent YouTube tutorial for this project at http://bit.ly/2kdGX8d
This project is a condensed version of his more extensive (and superior) work
'''

from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import cv2
import pickle
import glob
from tracker import tracker

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "camera_cal/calibration__pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def image_only_yellow_white(image):
    # setup inRange to mask off everything except white and yellow
    lower_yellow_white = np.array([140, 140, 64])
    upper_yellow_white = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower_yellow_white, upper_yellow_white)
    return cv2.bitwise_and(image, image, mask=mask)

## Define a function that applies Gaussian Noise kernel
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

## Define a function that applies Canny transform
def canny(img, low_threshold, high_threshold, kernel_size):
    img = image_only_yellow_white(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = gaussian_blur(gray, kernel_size)
    return cv2.Canny(blur_gray, low_threshold, high_threshold)

# create a region of interest mask
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Useful functions for producing the binary pixel of interest images to feed into the LaneTracker Algorithm
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

def window_mask(width, height, img_ref, center,level):
		output = np.zeros_like(img_ref)
		output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
		return output


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
	return cv2.addWeighted(initial_img, α, img, β, λ) 


def process_image(image):

	# undistort the image
	img = cv2.undistort(image,mtx,dist,None,mtx)
	# mask calculations
	imshape = img.shape
	
	# process image and generate binary pixel of interests
	preprocessImage = np.zeros_like(img[:,:,0])
	gradx = abs_sobel_thresh(img, orient='x', thresh=(25,255)) # 12
	grady = abs_sobel_thresh(img, orient='y', thresh=(10,255)) # 25
	c_binary = color_threshold(img, sthresh=(100,255), vthresh=(200,255)) 
	preprocessImage[((gradx == 1) & (grady == 1) | (c_binary == 1) )] = 255

	# Order of array: bottom_x1, bottom_y1; top_x1, top_y2; top_x2, y_top; bottom_x2, bottom_y2
	vertices = np.array([[(0, imshape[0]), (imshape[1]*14/32, imshape[0]*9/16), (imshape[1]*16/32, imshape[0]*9/16), (imshape[1], imshape[0])]], dtype=np.int32)
	masked_edges = region_of_interest(preprocessImage, vertices)

	# work on defining perspective transformation area
	img_size = (img.shape[1],img.shape[0])
	bot_width = .76 # percent of bottom trapizoid height
	mid_width = .1 # percent of middle trapizoid height
	height_pct = .63 # percent for trapizoid height -- sets extent of range down the road
	bottom_trim = .935 # percent from top to bottom to avoid car hood 

	src = np.float32([[img.shape[1]*(.5-mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(.5+mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(.5+bot_width/2),img.shape[0]*bottom_trim],[img.shape[1]*(.5-bot_width/2),img.shape[0]*bottom_trim]])
	offset = img_size[0]*.25
	dst = np.float32([[offset, 0], [img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]], [offset ,img_size[1]]])

	# perform the transform
	M = cv2.getPerspectiveTransform(src,dst)
	Minv = cv2.getPerspectiveTransform(dst,src)
	warped = cv2.warpPerspective(masked_edges,M,img_size,flags=cv2.INTER_LINEAR)

	window_width = 25
	window_height = 80

	# establish overall class to do tracking
	curve_centers = tracker(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = 25, My_ym = 10/720, My_xm = 4/500, Mysmooth_factor = 15)	

	window_centroids = curve_centers.find_window_centroids(warped)

	# Points to draw all left and right windows
	l_points = np.zeros_like(warped)
	r_points = np.zeros_like(warped)

	# Points to find left and right lanes
	rightx = []
	leftx = []

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

	# Draw the results
	template = np.array(r_points+l_points,np.uint8) # add l, r window pixels
	zero_channel = np.zeros_like(template) # create zero color channel
	template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8)# make window pixels green
	warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # make original road pixels 3 color channels
	result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the original road image with window results

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
	inner_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2, right_fitx[::-1]-window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

	# draw lane lines, middle curve, road background on two different blank overlays
	road = np.zeros_like(img)
	road_bkg = np.zeros_like(img)
	cv2.fillPoly(road,[left_lane],color=[255, 0, 0])
	cv2.fillPoly(road,[right_lane],color=[0, 0, 255])
	cv2.fillPoly(road,[inner_lane],color=[0, 255, 0])
	cv2.fillPoly(road_bkg,[left_lane],color=[255, 255, 255])
	cv2.fillPoly(road_bkg,[right_lane],color=[255, 255, 255])
	
	road_warped = cv2.warpPerspective(road,Minv,img_size,flags=cv2.INTER_LINEAR)
	road_warped_bkg = cv2.warpPerspective(road_bkg,Minv,img_size,flags=cv2.INTER_LINEAR)

	base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
	result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)

	ym_per_pix = curve_centers.ym_per_pix # meters/pixel in y dimension
	xm_per_pix = curve_centers.xm_per_pix # meters/pixel in x dimension
	curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix, 2)
	curverad = ((1 + (2*curve_fit_cr[0]*yvals[1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) /np.absolute(2*curve_fit_cr[0])

	# calculate the offset of the car on the road
	camera_center = (left_fitx[-1] + right_fitx[-1])/2
	center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
	side_pos = 'left'
	if center_diff <= 0:
		side_pos = 'right'

	cv2.putText(result,'Radius of Curvature = '+str(round(curverad,3))+'(m)',(50,50) , cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255),2)
	cv2.putText(result,'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center',(50,100) , cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255),2)

	return result

Output_video = 'output1_tracked.mp4'
Input_video = 'project_video.mp4'
# Input_video = 'harder_challenge_video.mp4'

clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_image) 
video_clip.write_videofile(Output_video, audio=False)


# write_name = './test_images/road_warped'+str(idx)+'.jpg'
# cv2.imwrite(write_name, result)


