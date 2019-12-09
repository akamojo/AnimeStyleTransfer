import PIL
from PIL import Image
import os
import sys

width, height = int(sys.argv[1]), int(sys.argv[1])
resized_dir = f"{width}x{height}"
if not os.path.isdir(resized_dir):
	os.mkdir(resized_dir)
	
files_list = os.listdir()
for idx, filename in enumerate(files_list):
	if filename.endswith('.jpg'):
		try:
			with Image.open(filename) as im:
				im_resized = im.resize((width, height), PIL.Image.LANCZOS)
				im_resized.save(os.path.join(resized_dir, filename), "JPEG")
				if idx % 25 == 0:
					print(f"{idx} / {len(files_list)} processed")
		except IOError:
			print(f"{idx}. Problems reading file {filename}")
print("Done!")
		