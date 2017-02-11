
'''
With great thanks to Udacity Content Developer Aaron Brown for his excellent YouTube tutorial for this project at http://bit.ly/2kdGX8d
This project is a condensed version of his more extensive project
'''

import numpy as np
import cv2
class tracker():

    # when starting a new instance please be sure to specify all unassigned variables 
    def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym = 1, My_xm = 1, Mysmooth_factor = 15):
        # list that stores all the past (left,right) center set values used for smoothing the output 
        self.recent_centers = []

        # the window pixel width of the center values, used to count pixels inside center windows to determine curve values
        self.window_width = Mywindow_width

        # the window pixel height of the center values, used to count pixels inside center windows to determine curve values
        # breaks the image into vertical levels
        self.window_height = Mywindow_height

        self.margin = Mymargin

        self.ym_per_pix = My_ym # meters/pixel in vertical axis

        self.xm_per_pix = My_xm # meters/pixel in horizontal axis

        self.smooth_factor = Mysmooth_factor

        # main tracking function for finding, storing lane segment positions
    def find_window_centroids(self, warped):

        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin

        window_centroids = [] # Store l, r window centroid positions each level
        window = np.ones(window_width) # Create window template for use with convolutions

        # Sum quarter bottom of image to get slice
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)

        # Add determinations for first layer
        window_centroids.append((l_center,r_center))

        # search each layer for max pixel locations
        for level in range(1,(int)(warped.shape[0]/window_height)):
            # Convolve the window into the verticle slice of the image
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)

            # Use past left center as ref. to find best left centroid
            # window_width/2 as offset as convolutional signal ref. is at right side of window, not center
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin, 0))
            l_max_index = int(min(l_center+offset+margin, warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # and again for right side
            r_min_index = int(max(r_center+offset-margin, 0))
            r_max_index = int(min(r_center+offset+margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # add result derived for that layer
            window_centroids.append((l_center, r_center))

        self.recent_centers.append(window_centroids)
        # return averaged values of line centers
        return np.average(self.recent_centers[-self.smooth_factor:], axis = 0)      