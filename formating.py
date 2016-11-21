from PIL import Image
from resizeimage import resizeimage



for i in range(12500):
 if(i % 500 == 0):
  print('i: ', i, '\n')
 cat = open("cat.{}.jpg".format(i), 'r+b')
 dog = open("dog.{}.jpg".format(i), 'r+b')
 cat_image = Image.open(cat)
 dog_image = Image.open(dog)
 cat_cover = resizeimage.resize_cover(cat_image, [128,128], validate=False)
 dog_cover = resizeimage.resize_cover(dog_image, [128,128], validate=False)
 cat_cover.save("cat_scaled.{}.jpg".format(i), cat_image.format)
 dog_cover.save("dog_scaled.{}.jpg".format(i), dog_image.format)
