# ####################
# Step 1, nef to tiff
# ####################

import numpy as np 
import sys, rawpy, imageio, glob, os


def nef2tif(nef, tiff_dir, is_save=True):		# rgb chennel problem, color change to red
	raw = rawpy.imread(nef)
	rgb_img = raw.postprocess(output_bps=8).copy()		# If you need differen light for surface/scatter, give a switch			use_camera_wb=True, bright=1.0,
	print(nef, np.max(rgb_img))

	if(is_save):
		tiff_name = nef.split('/')[-1].split('.')[0] + ".tiff"
		imageio.imsave(os.path.join(tiff_dir, tiff_name), rgb_img)
	return rgb_img

if __name__ == '__main__':
	if(len(sys.argv) == 1):
		source_dir = "./"
	elif(len(sys.argv) == 2):
		source_dir = sys.argv[1]
	else:
		print("Put this script under the block folder OR give the block folder name!")
		quit()
	source_nef = glob.glob(os.path.join(source_dir, '*_scatter.nef'))
	tiff_dir = os.path.join(source_dir,"tiff")

	if(not os.path.exists(tiff_dir)):
		os.mkdir(tiff_dir)
		print("Created {}".format(tiff_dir))

	for nef in source_nef:
		tiff_img = nef2tif(nef, tiff_dir)

