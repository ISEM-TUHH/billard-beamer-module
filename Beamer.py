"""The Beamer Module for the Billard@ISEM system.

This module supplies the class `Beamer` and the function `display_image`. `Beamer` supplies the webserver and API, while `display_image` hosts a `cv2.imshow` thread, actually showing the image.

Attributes:
	update_frame_flag (bool): if true, the running `display_image` instance will show the currently defined `frame`
	frame (np.ndarray): cv2 image that gets shown
	TEST_MODE (bool): gets set by the running `Beamer` instance. Corresponds to `Module.TEST_MODE`. If False, the image will be displayed in fullscreen mode.
	END_DISPLAY (bool): if True, the running `display_image` instance will terminate.

"""


from billard_base_module.Module import Module
from billard_base_module.RemoteModules import Camera

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

update_frame_flag = True
frame = None
TEST_MODE = True
END_DISPLAY = False

class Beamer(Module):
	"""Implementation of the Beamer Module for the Billard@ISEM system.

	Running it also requires the `display_image` function two run in a parallel thread.´ to handle actually displaying the image.

	Attributes:
		current_dir (str): Absolute path of this file
		app (flask.Flask): Flask app. This needs to run for the module to run (e.g., `beamer.app.run()`)
		passUnitMatrixWarning (bool): if True, you can save a unit matrix as the transformation matrix
		M (np.ndarray 3x3): perspective warp transformation matrix for transforming the image using `cv2`.
		trace_type (str): type of trace that gets drawn in a live inference type mode (see Beamer.put_white_points)
		trace_length (int): how long the trace history should be (see Beamer.put_white_points)
		trace_history (list): the last `Beamer.trace_length` trace lists (see Beamer.put_white_points)
		available_sounds (list): list of all .mp3 files in the `sounds` directory (including subdirectories)
		state (str): current state of the Beamer Module. Not really used
		black_image (np.ndarray): a 1080x1920 (full HD) black image
	"""


	def __init__(self, config="config/config.json", template_folder="templates"):
		"""Initialize a Beamer instance.

		This collects all sound resources and uses the parent class `Module` to setup the web API.

		Args:
			config (str, optional): Relative path to a config (`.json`) file. Defaults to "config/config.json".
			template_folder (str, optional): Relative path to a folder to used as templates by flask. Defaults to "templates".
		"""
		global TEST_MODE

		current_dir = os.path.dirname(__file__)
		self.current_dir = current_dir
		Module.__init__(self, config=f"{current_dir}/{config}", template_folder=f"{current_dir}/{template_folder}", static_folder=f"{current_dir}/static")

		self.camera = Camera(self.getModuleConfig("camera"))

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
				#"config": self.configure_output,
				"updateconfigimage": self.update_config_image,
				"camin": self.configure_camera_answer,
				"manin": self.configure_manual_answer,
				"state": self.get_state,
				"transform": self.do_transform,
				"savematrix": self.save_transformation_matrix,
				"overwritesafety": self.overrule_warning
			},
			"debug": {
				#"control": self.control_image,
				"relaunch": self.force_restart,
				"currentimage": self.current_image
			},
			"config": {
				"camera_calibration": self.start_camera_calibration,
				"manual_calibration": self.configure_output
			}
		}
		self.add_all_api(api_dict)

		self.state = "free"
		self.black_image = np.zeros((1080,1920))


	def index(self):
		"""Renders the index website.

		Returns:
			Response: rendered index.html page
		"""
		print(f"Client connected.")

		self.state = "home"
		return render_template('index.html')

	def get_state(self):
		""" Returns the current state of the beamer object.

		Used to track the current task globally. Not really useful.

		Returns:
			Respone: `flask.jsonify` output of `{'state': self.state}`
		"""
		return jsonify({"state": self.state, "comment": "The state is not really used anywhere."})

	def play_sound(self):
		"""Plays a sound using mpg123

		From a flask request.json, search for the sound behind the json key `sound` in `self.available_sounds` and plays it using mpg123. 

		Known errors: when using a Raspberry PI 5 connected via HDMI to a beamer and turing the beamer on after the PI, there sometimes is no sound.

		Returns:
			str, (int): "Playing [file name]" if everythin is fine or a message and 404 when the sound was not found.
		
		"""
		res = request.json
		cleaned = [x.split("/")[-1].replace(".mp3", "") for x in self.available_sounds]
		if "sound" in res.keys() and res["sound"] in cleaned:
			file = self.available_sounds[cleaned.index(res["sound"])]
			print("playing sound", file)
			subprocess.Popen(["mpg123", file], stdout=subprocess.PIPE) # block stdout (pipe into nirvana)
			return "Playing " + file
		else:
			return f"Requested sound '{res}' not found or bad request json", 404

	def sound_volume(self):
		"""Change the volume level for sound replay using amixer
		
		From a POST request, use the json data `{"level": 100}` to set the volume between 0 and 100 using the `amixer` utility. The key `level` not in the json data or the level being outside of 0 <= level <= 100 returns in a 403 error. 

		Returns:
			str, int: "" and the http status code 200 if everything worked or 403 if there is something wrong with the request. 
		"""
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
		"""This API endpoint receives images from the web (game module), transforms and shows it.

		POST an image (as the request.data) enocded to uint8.

		Returns:
			str: "Image received", flask requires response
		"""
		res = request
		#image = res["media"]
		#print(res.data)
		self.frame = cv2.imdecode(np.frombuffer(res.data, np.uint8), cv2.IMREAD_COLOR)
		self.update_frame(self.frame)

		return "Image received"

	def put_white_points(self):
		"""Receive moving balls (with a flask POST request) as json data in the format {"points": [{"x": 123, "y": 345}, ...]} and place them on the canvas.

		Coordinates are individually transformed, improving performance compared to drawing them and then transforming the entire image. This leads to small projection errors when drawing stuff around transformed coordinates.
		Stores the last `self.trace_length` (transformed) coordinates in `self.trace_history` (queue like behaviour).

		The type of visual affect must be added via code and could be changed by manipulating `self.trace_type`. Current "doppler" effect displays squares increasing in size and decreasing in brightness by the "age" of the coordinates, simulating the physical doppler effect.

		Returns:
			str: confirmation "Displaying moving balls"

		Todo:
			- Improve determining the size of the ball
			- Add trail of previous balls?
			- Add more effects?
		"""
		global frame, update_frame, transformed_frame

		# transformation of individual points/coordinates is way faster than drawing on the image and transforming the entire image. The warping should be negligible.
		# Transformation procedure is form https://stackoverflow.com/questions/36584166/how-do-i-make-perspective-transform-of-point-with-x-and-y-coordinate 
		res = request.json

		if len(res["points"]) == 0:
			return "No balls send to display, json key 'points' is empty"

		src = np.array([[x["x"], x["y"]] for x in res["points"]], dtype="float32")
		src2 = src * np.array([1920/2230, 1080/1115])
		pts = np.array([src2], dtype="float32")

		if self.config["transformation"] == "OpenCV":
			transformed = cv2.perspectiveTransform(pts, self.M)[0]
		else:
			transformed = src
			
		canvas = transformed_frame.copy() # != self.frame, global frame is already transformed

		#transformed = cv2.perspectiveTransform(pts, self.M)[0]
		self.trace_history.insert(0, transformed)
		if len(self.trace_history) > self.trace_length:
			self.trace_history.pop(-1)

		# the type of drawing is determined by self.trace_type. Currently only "doppler" is implemented.
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
		"""Display a black image on the beamer

		Returns:
			str: "Displaying black image", flask requires a response
		"""
		global transformed_frame
		self.update_frame(self.black_image, do_transformation=False) # when using OpenCV for transformation, this is way faster without transformation
		transformed_frame = self.black_image.copy() # for nicer fast inference
		return "Displaying black image"


	###################### Setup/Calibration methods ####################################
	def do_transform(self, getFromFile = True):
		"""Transform the output image based on the transformation matrix `self.M` or read from the file behing `self.transformPath`. 

		The type of transformation is determined by `'transformation'` in the config file. Setting it to `'OpenCV'` manually transforms the image using the OpenCV (cv2) library before showing it. `'xrandr'` uses the xrandr system to transform the entire desktop output (this is untested and likely not working!).

		Args:
			getFromFile (bool, optional): Decide if the current `self.M` should be used or if the matrix should be loaded from `self.transformationPath` file. Defaults to True.

		Returns:
			str: description of the transformation process.
		"""
		M = self.M

		if getFromFile:
			with open(self.transformPath, "r") as file:
				transTotal = json.load(file)
				M = np.array(transTotal["transformation-matrix"])
				#print(M, M.dtype)

		self.state = "configured"
		self.M = M

		match self.config["transformation"]:
			case "xrandr":
				transform_argument = ",".join(M.flatten().astype(str))
				display = self.config["dp-name"] # get the name of the current display for the xrandr call
				int_dim = self.config["internal-dimensions"]
				int_width, int_height = int_dim["width"], int_dim["height"]
				#mes = transform_list#"Process was not called."
				mes = os.popen(f'xrandr --output {display} --transform {transform_list}').read() # https://x.org/releases/X11R7.5/doc/man/man1/xrandr.1.html

			case "OpenCV":
				self.update_frame(self.frame) # reload the image
				display = "(irrelevant)"
				mes = "transformed "
		
		return f"Transformed {display} with matrix: <br>{M}<br><br>Message from the process:<br> {mes}".replace("\n","<br>")
	
	def overrule_warning(self):
		self.passUnitMatrixWarning = True
		return "You have overwritten the safety check to not write an unit matrix as the transformation matrix."

	def save_transformation_matrix(self):
		if np.array_equal(self.M, np.eye(3)) and not self.passUnitMatrixWarning:
			return "WARNING! You are close to overwriting the matrix with an unit matrix. If this is not on purpose, dont send this signal again. If you want to write an unit matrix, call /v1/overwritesafety first."

		with open(self.transformPath, "r+") as file:
			transTotal = json.load(file)
			transTotal["transformation-matrix"] = self.M.tolist()
			asStr = json.dumps(transTotal, indent=4)

			file.seek(0)
			file.write(asStr)
			file.truncate()
		self.passUnitMatrixWarning = True
		return f"Written transform.json: <br> {asStr}".replace("\n","<br>")

	def configure_output(self):
		"""Start the manual config process 
		
		This starts the routine to align the beamer output to the pool table as it is recognised by the camera. Displays the configuration image and pings the camera module to measure and respond. (Last not yet implemented)

		Currently, this just displays a grid on the beamer for manual configuration.

		Returns:
			Response: rendered configure.html template
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

		self.update_frame(self.frame)
		return render_template("configure.html")

	def update_config_image(self):
		"""In the manual config process, this puts live updates from the website (passed as "corner" key in request.json) on the projected image. This allows for the user doing the calibration to see where the corner they selected is.

		Returns:
			str: "Top", flask requires a response
		"""
		res = request.json
		print(res)
		self.corners[res["corner"]] = res
		print(self.corners)
		self.frame = self.config_raw_frame.copy()
		for c in self.corners:
			#print(c, self.corners[c])
			self.frame = cv2.drawMarker(self.frame, (int(self.corners[c]["x"]), int(self.corners[c]["y"])), (0,0,0), cv2.MARKER_CROSS, markerSize=15, thickness=5)

		self.update_frame(self.frame)

		return "Top"

	def configure_manual_answer(self):
		""" 
		
		Process the answer of the client when the configuration is in manual mode
		
		Returns:
			str: html message containing the points and transformation matrix.
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
		"""NOT WORKING part of trying to calibrate the beamer based on the Camera Module detecting the beamer.

		This function has been mostly generated by ChatGPT-4.1 on 11.2.26.
		
		Returns:
			Response: an image/jpg mimetype response of the transformed image
		"""
		#print(request.json) # -> also returns the dimensions of the table
		#w, h = 2150, 1171
		args = request.args
		print(args)


		width, height = 1920, 1080
		rect_w, rect_h = 0.3 * width, 0.3 * height
		left = 0.5 * (width - rect_w)
		top = 0.5 * (height - rect_h)

		x1 = float(args.get("x1"))
		y1 = float(args.get("y1"))
		x2 = float(args.get("x2"))
		y2 = float(args.get("y2"))
		x3 = float(args.get("x3"))
		y3 = float(args.get("y3"))
		x4 = float(args.get("x4"))
		y4 = float(args.get("y4"))

		corners_cam = np.array([
			[x1, y1],
			[x2, y2],
			[x3, y3],
			[x4, y4]
		])
		corners_b = np.array([
			[0.35, 0.35],
			[0.65, 0.35],
			[0.35, 0.65],
			[0.65, 0.65]
		]) * np.array([1920, 1080])

		self.M = cv2.getPerspectiveTransform(corners_cam.astype(np.float32), corners_b.astype(np.float32))
		self.do_transform(getFromFile=False)
		return f"Calculated transformation matrix {self.M}<br>Check if correct, then save."


	def start_camera_calibration(self):
		"""NOT WORKING part of trying to calibrate the beamer based on the Camera Module detecting the beamer.
		
		Returns:
			Response: an image/jpg mimetype response of the transformed image

		"""

		img = cv2.imread("storage/calibration.png")

		self.frame = img
		_, buffer = cv2.imencode(".jpg", img)
		self.update_frame(img, do_transformation=False)
		self.M = np.eye(3)
		#self.do_transform(getFromFile=False)
		#return Response(buffer.tobytes(), mimetype="image/jpg")

		requests.get(self.camera.endpoint("/v1/startbeamercalibration"))
		print("Started camera calibration")
		return render_template("cameraConfig.html", camera_video_feed=self.camera.endpoint("/website/video_feed"), camera_stop=self.camera.endpoint("/v1/stopgeneration"))

	def current_image(self):
		_, buffer = cv2.imencode(".jpg", self.frame)
		return Response(buffer.tobytes(), mimetype="image/jpg")
		


	def force_restart(self):
		"""Stop the system by ending both threads. If this is running on a systemd service that restarts a finished service, this restarts the server. Useful for debugging.

		Returns:
			str: "Restarting", because flask requires a response
		"""

		global END_DISPLAY
		END_DISPLAY = True
		os.kill(os.getpid(), signal.SIGINT)
		return "Restarting"

	
	###################### INTERACTION WITH GUI THREAD #################################
	def update_frame(self, new_frame, do_transformation=True):
		"""Pass a new frame to be displayed in the display thread.

		From the Beamer object, this updates the global variables `frame`, `update_frame` and `transformed_frame`. The first two signal the `display_image` function running in a different thread to update the displayed image.

		If the new frame should be perspectively transformed before displaying, supply `do_transformation=True`. This also saves the last transformed image in the global variable `transformed_frame`. 

		Args:
			new_frame (np.ndarray): new cv2 image to get displayed
			do_transformation (bool, optional): Decide if the image should be transformed before displaying. Only effective if `'transformation': 'OpenCV'` in the used config file. Defaults to True.
		"""

		global frame, update_frame_flag, transformed_frame#, opencv_transformation
		dim = self.config["beamer-dimensions"]
		width, height = dim["width"], dim["height"]
		#print("Send image with id ", self.last_frame_timestamp)
		#print("Updating image...")
		# scale down the image so it always fills out the beamer perfectly
		# with the images not being optimised for the table (roughly 2:1) but the beamer being 16:9 (and its transformation matrix M being calculated for 1920x1080), we need to temporarely squish the image.
		if do_transformation and (self.config["transformation"] == "OpenCV"):
			scaledImage = cv2.resize(new_frame, (width, height))
			img = cv2.warpPerspective(scaledImage, self.M, (width, height))
			transformed_frame = img
		else:
			img = new_frame
			#transformed_frame = img

		frame = img
		update_frame_flag = True
		return 

def display_image():
	"""This function runs in its own thread, running cv2.imshow to display the global variable `frame` (np.ndarray / cv2 image). As soon as the flag `update_frame_flag` (bool) is raised, it updates the frame, resetting `updating_frame=False`. If the flag `END_DISPLAY` is raised, this function (the `while True` loop) terminates and closes all windows.

	In the test mode (flag ´TEST_MODE` in python, externally set with environment variable `PROD_OR_TEST=TEST`), the image is not displayed in fullscreen. 
	"""
	global frame, update_frame_flag, TEST_MODE, END_DISPLAY
	if not TEST_MODE:
		cv2.namedWindow("beamer", cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty("beamer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	#print("DISPLAYING NEW IMAGE")
	while True:
		if update_frame_flag and frame is not None :
			#print("UPDATING FRAME")
			cv2.imshow("beamer", frame)
			if cv2.waitKey(10) & 0xFF == ord('q'):  # Allow quitting with 'q' key
				break
			update_frame_flag = False

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