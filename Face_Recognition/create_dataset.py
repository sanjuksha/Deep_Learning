
import random
import os
import cv2
import errno
def make_sure_path_exists(path):
	try:
		os.makedirs(path)								
	except OSError as exception:
		if exception.errno!=errno.EEXIST:
			raise
	return path
#Create a test and train folder. That is not auto
data_directory='E:/Deep Learning/Homework 7-8/HW7DATA/DATA_PHOTOS'
train_directory='E:/Deep Learning/Homework 7-8/DATASET/train'
validation_directory='E:/Deep Learning/Homework 7-8/DATASET/validation'
num_validation=3000
random.seed(0)
list1=os.listdir(data_directory)

for f in list1:
	path=data_directory+"/"+f
	class_train_path=make_sure_path_exists(train_directory+'/'+f)
	class_validation_path=make_sure_path_exists(validation_directory+'/'+f)
	filenames=os.listdir(path)
	random.shuffle(filenames)
	training_filenames=filenames[num_validation:]
	validation_filenames=filenames[:num_validation]
	for name in validation_filenames:
		destination_image_path=class_validation_path+'/'+name
		img=cv2.imread(os.path.join(path,name))
		cv2.imwrite(class_validation_path+'/'+name,img)

	for name2 in training_filenames:
		destination_image_path=class_train_path+'/'+name2
		img=cv2.imread(os.path.join(path,name2))
		cv2.imwrite(class_train_path+'/'+name2,img)








