Object tracking an algorithm based on dlib correlation tracker, used to automatically detect track objects specified. The algorithm excepts a video file to first detect and once detected it automatically start tracking it   
Requirements:   YoloV4 weights for object detection - https://drive.google.com/u/0/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

# Object Detection
The object detection is done via darknet.We used https://github.com/AlexeyAB/darknet and cloned Darknet repo. Enable changes in the makefile  as per your system ad make sure to enable ".so" to use Darknet python API's via libdarknet.so. Once done Download the Yolov4 weights mentioned in the link above and use the python api's to do detection over images.

# Dlib Co-relation Tracking
The dlib correlation tracker implementation is based on Danelljan et al.’s 2014 paper, Accurate Scale Estimation for Robust Visual Tracking.Their work, in turn, builds on the popular MOSSE tracker from Bolme et al.’s 2010 work, Visual Object Tracking using Adaptive Correlation Filters. While the MOSSE tracker works well for objects that are translated, it often fails for objects that change in scale.

The work of Danelljan et al. proposed utilizing a scale pyramid to accurately estimate the scale of an object after the optimal translation was found. This breakthrough allows us to track objects that change in both (1) translation and (2) scaling throughout a video stream — and furthermore, we can perform this tracking in real-time.

We just need to provide the Bounding box co-ordinates  to the dlib tracker and the work is done. The code allows you to select the objects in the video you want to track.All the objects in coco.names can be tracked using the object Tracker. 

# Other Dependencies 
1. Imutils 
2. dlib
3. open-cv
4. Darknet
