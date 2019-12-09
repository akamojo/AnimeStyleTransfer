import PIL
from PIL import Image
import os

width, height = 32, 32 # TODO: move to cmd line arguments?
files_list = os.listdir()
for idx, filename in enumerate(files_list):
	if filename.endswith('.jpg'):
		try:
			with Image.open(filename) as im:
				im_resized = im.resize((width, height), PIL.Image.LANCZOS)
				im_resized.save(os.path.join(f'{width}x{height}', filename), "JPEG")
				if idx % 25 == 0:
					print(f"{idx} / {len(files_list)} processed")
		except IOError:
			print(f"{idx}. Problems reading file {filename}")
print("Done!")
		