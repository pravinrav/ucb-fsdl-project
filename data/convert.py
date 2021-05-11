import os
def convert():
	os.system("mkdir data")
	with open('val_annotations.txt', "r") as f:
		for e in f:
			line = e.split("\t")
			img, img_class = line[0], line[1]

			if not os.path.isdir("data/{0}".format(img_class)):
				os.system("mkdir data/{0}".format(img_class))
			os.system("cp images/{0} data/{1}/{0}".format(img, img_class))

convert()
