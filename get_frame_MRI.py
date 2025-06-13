import imageio
import numpy as np
import SimpleITK as sitk

import matplotlib.pyplot as plt
import sys, glob, os

if __name__ == "__main__":
	arguments = sys.argv[1:]

	if(len(arguments)<2):
		print("Provide the input dir, frame and output dir you need!")
		quit()

	elif(len(sys.argv) == 3):
		source_dir = sys.argv[1]
		idx_frame = int(sys.argv[2])
		output_dir = "./"

		exvivo_volume = sitk.ReadImage(source_dir, sitk.sitkFloat32)

		try:
			exvivo_slice = sitk.GetArrayFromImage(exvivo_volume)[int(idx_frame)-1, :, :]
			# rgb_exvivo_slice = np.stack((normalize_image(exvivo_slice),)*3, axis=-1)

			imageio.imsave(os.path.join(output_dir, f'{idx_frame}.tiff'), exvivo_slice)
			# np.save(os.path.join(output_dir, f'{idx_frame}.npy'), exvivo_slice)

		except Exception as e:
			raise e

	else:
		source_dir = arguments[0]
		idx_frame = arguments[1:-1]
		output_dir = arguments[-1]

		exvivo_volume = sitk.ReadImage(source_dir, sitk.sitkFloat32)

		try:
			for i in idx_frame:
				exvivo_slice = sitk.GetArrayFromImage(exvivo_volume)[int(i)-1, :, :]
				# rgb_exvivo_slice = np.stack((normalize_image(exvivo_slice),)*3, axis=-1)

				imageio.imsave(os.path.join(output_dir, f'{i}.tiff'), exvivo_slice)

		except Exception as e:
			raise e
