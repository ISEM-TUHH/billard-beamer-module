This is the implementation for a Beamer Module as defined in documentation.

To have the fully working program, you have to launch both Beamer.py (which provides the webserver) and GUI.py (which provides the local GUI). Due to threading problems in tkinter, they need to run in different main threads. They communicate via port 5000.