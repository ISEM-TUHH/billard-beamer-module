# Beamer module for the Billard@ISEM system
This is the implementation for a Beamer Module as defined in documentation.

Using a Beamer above the billard table, the system gains the ability to project images directly onto the table. A perspective warp can be applied after manual calibration for the projection to perfectly fit the table. If the system running this software is able to play sounds (like most commercial beamers), this module can also output sound and set the volume. 

Run `Beamer.py` with a connected beamer to start the webserver and local GUI/display thread.

## Installation and assumptions
- Hardware recommendations: A beamer connected to a small computer (like a Raspberry PI) running a linux version (like Raspberry PI OS). If the beamer has different resolution than 1080x1920, you will need to individually recreate the configuration image in `static/grid.bmp`.

1. Assumptions about the system you are installing this on: [`xrandr`](https://x.org/releases/X11R7.5/doc/man/man1/xrandr.1.html) and `git` are available. This is the case on most linux versions, especially Raspberry PI OS (only tried on a non-headless version). For debugging minor knowledge of `systemd`/`systemctl` and `journalctl` are beneficial.

2. Clone the repository with submodules: `git clone --recurse-submodules https://github.com/ISEM-TUHH/billard-beamer-module.git`
    - For ease of installation, clone it into your home directory (`/home/<username>/`)

3. Go into the newly created directories installation toolbox and run the installation script.
    - Edit `install/billard-beamer-server.service` to have the correct username (replace `beamer-pi`).
    - Edit `config/config.json` to have the correct informations for your setup. If you want to develop, also edit the `config/test_config.json` file.
    - `cd billard-beamer-module/install`
    - `sudo chmod +x install.sh && sudo ./install.sh` (sudo is required as this installs a new `systemctl` service)
        - This install and starts a new systemd service `billard-beamer-server.service`.

4. If successful, a test image should be displayed and the server should be availabe under port 5000 of the given IP address: `http://<address>:5000`
    - This website is very barebones and mostly used for configuration and testing

## Configuration
This part is concerned with setting up the display area to perfectly match the billard table.
1. Go to the modules configuration website under `http://<address>:5000/v1/config`
2. This displays a grid on the beamer. 
    - By selecting a corner on the website and clicking somewhere on the image, a marker is displayed at corresponding place on the beamer
    - Repeat until you perfectly match the corner you are wanting to fixate
        - If you are also using the camera module, these should be configured on the same corners: the most outer corners of the camera markers
    - Repeat for all corners
        - If some corners are not in the display area of the beamer, you can manually edit the pixel position of the corners on the website.
    - Click on the "Apply configuration" button to see your changes applied.
        - If you want to iterate, reload the page and repeat. Reloading the page resets the configuration
        - If you want to keep the configuration, click on "Save the configuration". You need to have a configuration applied previously for this to apply.
        