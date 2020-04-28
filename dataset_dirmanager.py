import os
import shutil

# Make new directory (this dir wil contain subdirs)
new_imagedir = "data_per_painter"
if new_imagedir not in os.listdir():
	os.mkdir(new_imagedir)

# Specify directory of images to be sorted per painter
imagedir = "data/resized/resized"

# Make set of painter names
painters = set()
for f in os.listdir(imagedir):
	filename, ext = os.path.splitext(f)
	painter_name = filename.rstrip("0123456789_")
	painters.add(painter_name)

for painter in painters:
	# make new directories for each painter
	painterdir = new_imagedir+"/"+str(painter)
	if painter not in os.listdir(new_imagedir):
		os.mkdir(painterdir)
		painterdir = painterdir+"/train"	
		os.mkdir(painterdir)

	# cp all files with name into the newly made dir with that name
	for image_filename in os.listdir(imagedir):
		if painter in image_filename:
			src = os.path.join(imagedir, image_filename)
			print(f"Copying {src} into {painterdir}")
			shutil.copy2(src, painterdir)
		
