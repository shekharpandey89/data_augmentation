# USAGE
# python augmentation.py --image cat.jpg --output output

# import the required packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse

# creating an object of the argument parse
ap = argparse.ArgumentParser()

# parsing the arguments
ap.add_argument("-k", "--image", required=True,
help="input image path which we want to augmented")
ap.add_argument("-l", "--output", required=True,
help="path of the directory where we save our augmented images")
ap.add_argument("-m", "--prefix", type=str, default="image",
help="output filename prefix")
args = vars(ap.parse_args())

# now we are going to load our input image on which we want to apply data augmentation,
# and then we convert it to a NumPy array, and after that we add an extra dimension to reshape
# it like for image classification
print("we are loading our image... ")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# we construct the object of the ImageDataGenerator
augObject = ImageDataGenerator(rotation_range=40,height_shift_range=0.2,
                               width_shift_range=0.2, shear_range=0.3, 
                               zoom_range=0.3,horizontal_flip=True, 
                               fill_mode="nearest")

totalImg = 0

# now we are going to construct the main Python generator
print("generating new images...")
imageDataGen = augObject.flow(image, batch_size=1, save_to_dir=args["output"],save_prefix=args["prefix"], save_format="jpg")

# loop over examples from our image data augmentation generator
for image in imageDataGen:
    # we are count our counter on per each iteration
    totalImg += 1

    # if we reached to the count == 10, then it will call break and stop because we are generating only 10 images
	# for demonstartion
    if totalImg == 10:
        break