# jaywalking

This model identifies pedestrians and traffic light states in real-time in order to capture jaywalking incidents in zoned crosswalks. It saves pictures of those jaywalking incidents for post-event analysis as well as an annotated video for replay. 

Please use the following steps to recreate the test results:

git clone git@github.com:jetwhale/jaywalking.git

Navigate to the root directory of the project
Run the following:

python3.13 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

Edit jaywalk_detector and input the test video you would like to analyze. The datasets I have included are night_example.mp4 and day_example.mp4.

The output video will be saved as the variable named VIDEO_OUT. The default for this variable is "output.mp4"

The output jaywalking incidents will be saved in a folder called `jaywalk_images` in the same directory as where the script is located. The folder will automatically be created if it's not present.
