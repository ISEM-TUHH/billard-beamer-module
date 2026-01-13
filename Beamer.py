from billard_base_module.Module import Module

import numpy as np
#import pandas as pd
import os
import subprocess
import glob
from flask import Flask, jsonify, render_template, request, Response
import socket
#import urllib.request
import requests
import json
import cv2
from PIL import Image, ImageDraw
import time
import io
import urllib
import logging
import threading
import signal

update_frame = True
frame = None
TEST_MODE = True
END_DISPLAY = False

class Beamer(Module):
	""" Implementation of the beamer module for the billard roboter
	"""

	def __init__(self, config="config/config.json", template_folder="templates"):
		global TEST_MODE

		current_dir = os.path.dirname(__file__)
		self.current_dir = current_dir
		Module.__init__(self, config=f"{current_dir}/{config}", template_folder=f"{current_dir}/{template_folder}", static_folder=f"{current_dir}/static")

		TEST_MODE = self.TEST_MODE

		self.transformPath = current_dir + "/storage/transform.json"
		self.M = np.eye(3) # neutral transformation matrix
		self.passUnitMatrixWarning = False
		# for the local GUI
		homescreenPath = current_dir + "/storage/homescreen.png"
		self.frame = cv2.imread(homescreenPath) # start out with a homescreen
		self.last_frame_timestamp = 0

		self.trace_type = "doppler"
		self.trace_length = 5
		self.trace_history = []

		self.do_transform()

		# Sound setup
		self.available_sounds = glob.glob("sounds/**/*.mp3", recursive=True)
		print("Available sounds:", self.available_sounds)

		print("Loaded config:", self.config)

		# GUI setup
		#cv2.namedWindow("beamer", cv2.WINDOW_NORMAL)
		#cv2.startWindowThread()
		#cv2.namedWindow("beamer", cv2.WINDOW_FULLSCREEN)
		#cv2.setWindowProperty("beamer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

		api_dict = {
			"": self.index,
			"v1": {
				"receiveimage": self.receive_image,
				"dynamicballs": self.put_white_points,
				"playsound": self.play_sound,
				"soundvolume": self.sound_volume,
				"off": self.display_black_image,
				"config": self.configure_output,
				"updateconfigimage": self.update_config_image,
				"camin": self.configure_camera_answer,
				"manin": self.configure_manual_answer,
				"state": self.state,
				"transform": self.do_transform,
				"savematrix": self.safe_transformation_matrix,
				"overwritesafety": self.overrule_warning
			},
			#"servegui": {
			#	"imagestamp": self.get_timestamp_last_image,
			#	"image": self.send_image
			#},
			"debug": {
				"control": self.control_image,
				"relaunch": self.force_restart
			}
		}
		self.add_all_api(api_dict)

		self.state = "free"
		self.black_image = np.zeros((1080,1920))


	def index(self):
		print(f"Client connected.")

		self.state = "home"
		return render_template('index.html')

	def state(self):
		""" Returns the current state of the beamer object.

		Used to track the current task globally
		"""
		return jsonify({"state": self.state})

	def play_sound(self):
		""" Plays a sound using mpg123 """
		res = request.json
		cleaned = [x.split("/")[-1].replace(".mp3", "") for x in self.available_sounds]
		if "sound" in res.keys() and res["sound"] in cleaned:
			file = self.available_sounds[cleaned.index(res["sound"])]
			print("playing sound", file)
			subprocess.Popen(["mpg123", file], stdout=subprocess.PIPE) # block stdout (pipe into nirvana)
			return "Playing " + file
		else:
			return f"Requested not sound '{res}' found or bad request json", 404

	def sound_volume(self):
		""" Change the volume level for sound replay using amixer """
		#print(request, request.json)
		res = request.json
		if "level" in res.keys():
			level = int(res["level"])
			if level < 0 or level > 100:
				return "", 403
			amixer_value = int(65536 * level/100)
			subprocess.Popen(["amixer", "set", "Master", str(amixer_value)], stdout=subprocess.PIPE)
			return "Volume set to " + str(level) + "%", 200
		return "", 403

	def receive_image(self):
		""" This API endpoint receives images from the web (game module), transforms and forwards them to the GUI.
		"""
		res = request
		#image = res["media"]
		#print(res.data)
		self.frame = cv2.imdecode(np.frombuffer(res.data, np.uint8), cv2.IMREAD_COLOR)
		self.update_frame(self.frame)

		return "Image received"

	def put_white_points(self):
		"""Receive moving balls (with a flask request) in the format {"points": [{"x": 123, "y": 345}, ...]} and place them on the canvas.

		Returns:
			str: confirmation "Displaying moving balls"

		Todo:
			- Improve determining the size of the ball
			- Add trail of previous balls?
		"""
		global frame, update_frame

		# transformation of individual points/coordinates is way faster than drawing on the image and transforming the entire image. The warping should be negligible.
		# Transformation procedure is form https://stackoverflow.com/questions/36584166/how-do-i-make-perspective-transform-of-point-with-x-and-y-coordinate 
		res = request.json

		if len(res["points"]) == 0:
			return "No balls send to display, json key 'points' is empty"

		src = np.array([[x["x"], x["y"]] for x in res["points"]], dtype="float32")
		src2 = src * np.array([1920/2230, 1080/1115])
		pts = np.array([src2], dtype="float32")

		transformed = cv2.perspectiveTransform(pts, self.M)[0]
		self.trace_history.insert(0, transformed)
		if len(self.trace_history) > self.trace_length:
			self.trace_history.pop(-1)

		canvas = transformed_frame.copy() # != self.frame, global frame is already transformed
		match self.trace_type:
			case "doppler":
				n_history = len(self.trace_history)
				for i, transformed in enumerate(self.trace_history):
					
					markerSize = 20*(i+1)
					color = np.array([255, 255, 255])*(1 - i/n_history)
					#print(color, type(color))
					for point in transformed:
						canvas = cv2.drawMarker(canvas, point.astype(int), color=color.astype(int).tolist(), markerType=cv2.MARKER_SQUARE, markerSize=markerSize) # always just a white circle. r=15 is a rough assumption


		self.update_frame(canvas, do_transformation=False)

		return "Displaying moving balls"

	def display_black_image(self):
		self.update_frame(self.black_image)
		return "Displaying black image"


	###################### Setup/Calibration methods ####################################
	def do_transform(self, getFromFile = True):
		M = self.M

		if getFromFile:
			with open(self.transformPath, "r") as file:
				transTotal = json.load(file)
				M = np.array(transTotal["transformation-matrix"])
				#print(M, M.dtype)

				# actually perform the xrandr call here
		#print(M)
		#print([list(x) for x in M])
		#transform_list = ",".join([",".join(list(x)) for x in M])
		transform_list = ""
		for x in M:
			for y in x:
				transform_list += str(y) + ","
		transform_list = transform_list[:-1]

		display = self.config["dp-name"] # get the name of the current display for the xrandr call
		int_dim = self.config["internal-dimensions"]
		int_width, int_height = int_dim["width"], int_dim["height"]
		mes = transform_list#"Process was not called."
		#mes = os.popen(f'xrandr --output "{display}" --fb {int_width}x{int_height} --transform {transform_list}').read()

		self.state = "configured"
		self.M = M
		self.update_frame(self.frame) # reload the image
		return f"Transformed {display} with matrix: <br>{M}<br><br>Message from the process:<br> {mes}".replace("\n","<br>")
	
	def overrule_warning(self):
		self.passUnitMatrixWarning = True
		return "You have overwritten the safety check to not write an unit matrix as the transformation matrix."

	def safe_transformation_matrix(self):
		if np.array_equal(self.M, np.eye(3)) and not self.passUnitMatrixWarning:
			return "WARNING! You are close to overwriting the matrix with an unit matrix. If this is not on purpose, dont send this signal again. If you want to write an unit matrix, "

		with open(self.transformPath, "r+") as file:
			transTotal = json.load(file)
			transTotal["transformation-matrix"] = self.M.tolist()
			asStr = json.dumps(transTotal, indent=4)

			file.seek(0)
			file.write(asStr)
			file.truncate()
		return f"Written transform.json: <br> {asStr}".replace("\n","<br>")

	def configure_output(self):
		""" This starts the routine to align the beamer output to the pool table as it is recognised by the camera. Displays the configuration image and pings the camera module to measure and respond. (Last not yet implemented)

		Currently, this just displays a grid on the beamer
		"""
		
		#load the old config
		#with open(self.transformPath, "r") as file:
		#	transTotal = json.load(file)
		#	self.M = np.array(transTotal["transformation-matrix"])
		self.M = np.eye(3) # init a new transformation matrix M
		#print(self.M)
		self.state = "configuring"

		gridPath = self.current_dir + "/static/grid.bmp"
		self.frame = cv2.imread(gridPath)
		self.config_raw_frame = self.frame
		self.corners = {} # tracks the clicked corners

		self.update_frame(self.frame)# just increase the counter
		return render_template("configure.html")

	def update_config_image(self):
		res = request.json
		print(res)
		self.corners[res["corner"]] = res
		print(self.corners)
		self.frame = self.config_raw_frame.copy()
		for c in self.corners:
			#print(c, self.corners[c])
			self.frame = cv2.drawMarker(self.frame, (int(self.corners[c]["x"]), int(self.corners[c]["y"])), (0,0,0), cv2.MARKER_CROSS, 7,3)
		self.update_frame(self.frame)# just increase the counter

		return "Top"

	def configure_manual_answer(self):
		""" Process the answer of the client when the configuration is in manual mode
		"""
		res = request.form
		print(res)

		dim = self.config["beamer-dimensions"]
		width, height = dim["width"], dim["height"]
		src_points = np.float32([[0,0],[width,0],[0,height],[width,height]])

		topLeft = [float(res["top-left-x"]),float(res["top-left-y"])]
		topRight = [float(res["top-right-x"]),float(res["top-right-y"])]
		botLeft = [float(res["bottom-left-x"]),float(res["bottom-left-y"])]
		botRight = [float(res["bottom-right-x"]),float(res["bottom-right-y"])]
		dst_points = np.float32([topLeft, topRight, botLeft, botRight])

		Ms = cv2.getPerspectiveTransform(src_points, dst_points)
		self.M = Ms @ self.M # allow for iterations
		print(self.M, "\n",Ms)
		res = self.do_transform(getFromFile=False)

		self.update_frame(self.frame)
		return f"Found this data:<br>{dst_points}<br><br>...and calculated the matrix:<br>{Ms}"

	def configure_camera_answer(self):
		""" This endpoint takes the answer of the camera (coordinates of the points projected in /config (python: configure_output) in camera coordinates) and performs the xrandr calibration and call.

		Currently in heavy development and using simulated/harcoded input.
		Not working in the moment, task for next sprint.
		"""
		#print(request.json) # -> also returns the dimensions of the table
		#w, h = 2150, 1171
		w, h = 1920, 1080 # res of the beamer
		wf, hf = 2150, 1171

		wscale = 5 # width of the calibration image/total width (defined by static/configuration-image) 
		src_points = np.float32([[0.25*wf,0.4*hf], [0.75*wf, 0.25*hf], [0.25*wf, 0.7*hf], [0.75*wf, 0.65*hf]]) # this will get parsed from the camera answer
		#src_points = np.float32([[0.25*w,0.4*h], [0.75*w, 0.2*h], [0.25*w, 0.7*h], [0.75*w, 0.65*h]])
		
		w5, h5 = w/5, h/5
		dst_points = np.float32([[2*w5,2*h5], [3*w5,2*h5], [2*w5,3*h5], [3*w5,3*h5]]) # these are correct!


		#T, mask = cv2.findHomography(dst_points, src_points, cv2.RANSAC, 5.0)
		T, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
		#T = cv2.getPerspectiveTransform(src_points, dst_points)
		#print(np.linalg.inv(T) - T2)

		# transform the corners of the camera/field in camera coordinates into the beamer system
		corners_field = np.array([[0,0,1],[wf,0,1],[0,hf,1],[wf,hf,1]]).T
		trans_dst = np.array((T @ corners_field)[[0,1], :].T, dtype="f4")
		corners_field = np.float32([[0,0],[wf,0],[0,hf],[wf,hf]])
		#trans_dst = cv2.perspectiveTransform(corners_field, T, (2,4))
		print("T", T)
		#print("trans_dst", trans_dst)


		# only for examples/debugging
		ew_src_points = np.float32([[0.25*wf,0.4*hf/5,1], [0.75*wf, 0.25*hf,1], [0.25*wf, 0.7*hf,1], [0.75*wf, 0.65*hf,1]]).T
		ew_trans_src = np.array(((T) @ ew_src_points)[[0,1], :].T, dtype="f4")
		#print(trans_dst, T, corners_field)
		#ew_trans_dst = cv2.perspectiveTransform(src_points, T, (2,4))
		ew_dst_points = np.float32([[2*w5,2*h5,1], [3*w5,2*h5,1], [2*w5,3*h5,1], [3*w5,3*h5,1]]).T
		ew_trans_dst = np.array((T @ ew_dst_points)[[0,1], :].T, dtype="f4")

		# completely transform it myself i guess?


		corners_beamer = np.float32([[0,0],[w,0],[0,h],[w,h]])

		print("src_points", src_points)
		print("dst_points", dst_points)
		M = cv2.getPerspectiveTransform(corners_beamer, trans_dst) # this also does exactly what its told

		#print(retval, mask)
		#print(T, trans_dst, M)
		#invM = np.linalg.inv(M)
		#print(invM)
		#scaledM = M * (np.array([1/wscale, 1/wscale, 1]))[:, None]

		img = cv2.resize(cv2.imread("debug/normal_image.jpg"), (w,h))
		#img = cv2.warpPerspective(img, M, (w,h))# this also does exactly what its told
		img = cv2.warpPerspective(img, T, (w,h))

		#cv2.imshow("transformed image", result)
		#cv2.waitKey(0)

		self.state = "configured"

		print("trans_dst", trans_dst)
		print("src_points", src_points)
		for c,u,m,i in zip(trans_dst, ew_trans_src, src_points, ew_trans_dst):
			#print(c, u,m,i)
			x, y = int(c[0]),int(c[1])
			u, z = int(u[0]),int(u[1])
			n, p = int(m[0]),int(m[1])
			a, b = int(i[0]),int(i[1])
			cv2.drawMarker(img, (x,y), (0,255,0), cv2.MARKER_CROSS, 15, 5) # green
			cv2.drawMarker(img, (u,z), (255,0,0), cv2.MARKER_CROSS, 15, 5) # blue
			cv2.drawMarker(img, (n,p), (0,0,255), cv2.MARKER_CROSS, 15, 5) # red
			cv2.drawMarker(img, (a,b), (0,255,255), cv2.MARKER_CROSS, 15, 5) # yellow

		_, buffer = cv2.imencode(".jpg", img)
		self.img = img
		return Response(buffer.tobytes(), mimetype="image/jpg")

		#return "This is a purely internal function."

	def control_image(self):
		""" In theory this should show what the camera sees of the beamer image, idk if its working :)
		"""
		wf, hf = 2150, 1171
		src_points = np.float32([[0.25*wf,0.4*hf], [0.75*wf, 0.25*hf], [0.25*wf, 0.7*hf], [0.75*wf, 0.65*hf]])
		w5, h5 = wf/5, hf/5
		dst_points = np.float32([[2*w5,2*h5], [3*w5,2*h5], [2*w5,3*h5], [3*w5,3*h5]]) # these are correct!

		M = cv2.getPerspectiveTransform(dst_points, src_points)#, dst_points)
		print(M)
		img = cv2.warpPerspective(self.img, M, (wf, hf))

		_, buffer = cv2.imencode(".jpg", img)
		return Response(buffer.tobytes(), mimetype="image/jpg")

	def force_restart(self):
		global END_DISPLAY
		END_DISPLAY = True
		os.kill(os.getpid(), signal.SIGINT)
		return "Restarting"

	
	###################### INTERACTION WITH GUI THREAD #################################
	def update_frame(self, new_frame, do_transformation=True):
		global frame, update_frame, transformed_frame
		""" Resize, transform and display a new frame """
		dim = self.config["beamer-dimensions"]
		width, height = dim["width"], dim["height"]
		#print("Send image with id ", self.last_frame_timestamp)
		#print("Updating image...")
		# scale down the image so it always fills out the beamer perfectly
		# with the images not being optimised for the table (roughly 2:1) but the beamer being 16:9 (and its transformation matrix M being calculated for 1920x1080), we need to temporarely squish the image.
		if do_transformation:
			scaledImage = cv2.resize(new_frame, (width, height))
			img = cv2.warpPerspective(scaledImage, self.M, (width, height))
			transformed_frame = img
		else:
			img = new_frame

		frame = img
		update_frame = True
		return 

def display_image():
	global frame, update_frame, TEST_MODE, END_DISPLAY
	if not TEST_MODE:
		cv2.namedWindow("beamer", cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty("beamer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	#print("DISPLAYING NEW IMAGE")
	while True:
		if update_frame and frame is not None :
			#print("UPDATING FRAME")
			cv2.imshow("beamer", frame)
			if cv2.waitKey(10) & 0xFF == ord('q'):  # Allow quitting with 'q' key
				break
			update_frame = False

		if END_DISPLAY:
			break

	cv2.destroyAllWindows()

if __name__ == "__main__":
	os.environ["DISPLAY"] = ":0"

	bea = Beamer()
	pid = os.getpid()
	print("PID", pid)
	#bea.configure_camera_answer()

	threading.Thread(target=display_image).start()

	# bea.app_run() creates two processes? (different PID)
	if bea.TEST_MODE:
		bea.app.run(host="0.0.0.0", port="5001")
	else:
		bea.app.run(host="0.0.0.0", port="5000")

	cv2.destroyAllWindows()