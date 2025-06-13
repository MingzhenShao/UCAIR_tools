# ####################
# Step 2, Crop with a input left-up & right-bottom point
# Before running this script, you should manually adjust some outlier.
# The crop box should be slightly bigger than what you want, 
# we'll crop again after the registration.
# ####################

import numpy as np 
import sys, imageio, glob, os
import cv2
from skimage import color


def crop_tissue(tiff, location, crop_dir, is_save=True, gray=False):  
	img = imageio.imread(tiff)

	if(gray):
		try:
			if(img.shape[2] == 4):
				img = color.rgba2rgb(img)
			if(img.shape[2] == 3):
				img = color.rgb2gray(img)
		except:
			pass

	W1 = location[0]
	H1 = location[1]
	W2 = location[2]
	H2 = location[3]
	
	if(is_save):
		crop_name = tiff.split('/')[-1].split('.')[0] + ".tiff"
		if(gray):
			imageio.imsave(os.path.join(crop_dir, crop_name), img[H1:H2, W1:W2])
		else:
			imageio.imsave(os.path.join(crop_dir, crop_name), img[H1:H2, W1:W2, :])
		
		print(os.path.join(crop_dir, crop_name))
	

if __name__ == '__main__':
	
	if(len(sys.argv) == 5):
		source_dir = "./"
		loc = []
		loc.append(int(sys.argv[1]))
		loc.append(int(sys.argv[2]))
		loc.append(int(sys.argv[3]))
		loc.append(int(sys.argv[4]))
	elif(len(sys.argv) == 6):
		source_dir = sys.argv[1]
		loc = []
		loc.append(int(sys.argv[2]))
		loc.append(int(sys.argv[3]))
		loc.append(int(sys.argv[4]))
		loc.append(int(sys.argv[5]))
	else:
		print("Put this script under the block folder OR give the block folder name!")
		print("Provide four values with order: x_left_up, y_left_up, x_right_bottom, y_right_bottom")	
		quit()

	source_tiff = glob.glob(os.path.join(source_dir, "tiff", "*.tiff"))		#"tiff", 
	
	crop_dir = os.path.join(source_dir, "crop")

	if(not os.path.exists(crop_dir)):
		os.mkdir(crop_dir)
		print("Created {}".format(crop_dir))

	
	for tiff in source_tiff:
		crop_img = crop_tissue(tiff, loc, crop_dir, True, False)




	
	