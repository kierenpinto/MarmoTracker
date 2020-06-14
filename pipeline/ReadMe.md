# Installation of dependancies
pipenv install
pipenv install pyrealsense2
pipenv install numpy
# Need to do this to allow CV2 to work 
$ cd ~/.virtualenvs/cv/lib/python3.6/site-packages/
$ ln -s /usr/local/python/cv2/python-3.6/cv2.so cv2.so