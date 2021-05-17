# Object-Tracking
Object tracking an algorithm based on dlib correlation tracker, used to automatically detect track objects specified. The algorithm excepts a video file to first detect and once detected it automatically start tracking it   
Requirements:   YoloV4 weights for object detection - https://drive.google.com/u/0/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

# Darknet
The object detection is done via darknet. Clone the Darknet and do changes in Makefile as per system configrations. Make sure to enable .so to use Darknet API's 

# Other Dependencies 
1. Imutils 
2. dlib
3. open-cv
