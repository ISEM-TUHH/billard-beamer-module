import tkinter as tk
import requests
from PIL import Image, ImageTk
from io import BytesIO
import os
import signal
import json
#import cv2 # for image transformation

class BeamerGUI:
    """ This class implements the local GUI for the beamer module.

    The communication is via a local api endpoint providing the image. 
    """
    def __init__(self):
        self.current_dir = os.path.dirname(__file__)
        with open(os.path.join(self.current_dir, "config/config.json"), "r") as f:
            self.config = json.load(f)
        
        if "DISPLAY" not in os.environ:
            os.environ["DISPLAY"] = ":0"
        
        self.root = tk.Tk()
        self.root.title("Image Viewer")
        self.root.attributes("-fullscreen", True)
        self.label = tk.Label(self.root)
        self.label.pack()

        self.last_timestamp = "1000acdc"
        self.wait_next_frame = 1000 # waittime until the next frame is loaded in ms

        self.update_image()


    def update_image(self):
        try:
            response = requests.get(f'http://localhost:5000/servegui/imagestamp')
            #print(response.json())

            # update the time to wait for the next request -> allow for higher framerates for e.g. videos.
            self.wait_next_frame = response.json()["next_request"]
            if response.json()["timestamp"] != self.last_timestamp:
                self.last_timestamp = response.json()["timestamp"]
                #print(str(response.text) == self.last_timestamp)
                print(response.text)

                image = requests.get("http://localhost:5000/servegui/image")

                img = Image.open(BytesIO(image.content))
                #img = img.resize((400, 400), Image.ANTIALIAS)  # Größe anpassen
                img = ImageTk.PhotoImage(img)
                self.label.config(image=img)
                self.label.image = img
            elif response.json()["timestamp"] != "restart":
                self.force_restart()
            else:
                pass
        except Exception as e:
            #print(f"Error fetching image: {e}")
            pass

        self.root.after(self.wait_next_frame, self.update_image)  # Alle 1 Sekunde aktualisieren

    def force_restart(self):
        """ This function kills the process (stops the server). This should restart the server, as it is listed in systemctl with restart=always
        """
        os.kill(os.getpid(), signal.SIGINT)
        return "Restarting the server."

if __name__ == '__main__':
    #root = tk.Tk()
    app = BeamerGUI()
    app.root.mainloop()
