from Module import Module
#import cv2 as cv
from PIL import Image
import math
import matplotlib.pyplot as plt
import numpy as np


class Beamer(Module):
	def __init__(self, id, res_x, res_y, template_folder=""):
		Module.__init__(self, id, template_folder=template_folder)
		self.x = res_x
		self.y = res_y
		self.radius = 10 # px

	def pic_pos(self):
		"""generates picture of resolution x times y with colored balls from camera module"""
		return

	def __find_data(self, source, target):
		"""Calculate transformation matrix for perspective transform for PIL.Image.transform()

		ripped from https://www.tutorialspoint.com/python_pillow/python_pillow_applying_perspective_transforms.html
		source: [[x,x],[x,x],[x,x],[x,x]] (int) coordinates in the original image which should be changed
		target: [[x,x],[x,x],[x,x],[x,x]] (int) coordinates where source pixels should be mapped to, everything inbetween will get transformed accordingly
		returns: np.array
		"""
		matrix = []
		for s, t in zip(source, target):
			matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
			matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
		A = np.matrix(matrix, dtype=float)
		B = np.array(source).reshape(8)
		res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
		return np.array(res).reshape(8)


	def transform_corner(self, image, corners_new, corners_old=None):
		"""perform a perpective distortion on image, where old corners get mapped on new corners
		
		image: PIL Image
		corners_old (optional): [(x,y),(x,y),(x,y),(x,y)], old corners of the image, default is corners of PIL Image in the order top left, top right, bottom left, bottom right.
		corners_new: like above, new corners to map to

		return: distorted PIL Image
		"""
		w, h = image.size
		if w > self.x or h > self.y:
			print("Transform Corner in Beamer: input image has a higher resolution than the beamer as claimed during init")
		if corners_old == None:
			corners_old = [(0,0),(w-1,0),(0,h-1),(w-1,h-1)]
		print(corners_old)
		M = self.__find_data(corners_old,corners_new)
		print(M)
		result = image.transform((self.x,self.y), Image.PERSPECTIVE, M)
		return result

if __name__ == "__main__":
	bea = Beamer("beamer", 1000, 500)
	img = Image.open("test.jpg")
	img2 = bea.transform_corner(img, [(100, 100), (500, 0), (0, 498), (523, 498)])
	img2.show()

