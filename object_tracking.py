import os

import cv2
import dlib
from imutils.video import FPS

import config as cfg
from object_detector import OBJECT_DETECTOR


class OBJECT_TRACKING:
    def __init__(self, input_vidfile: str, tracker: str = 'dlib', tracking_obj='truck'):
        self.vid_file = input_vidfile
        self.tracker = tracker
        self.track_obj = tracking_obj
        # initialize the variables
        self.initBB = None
        self.fps = None
        self.tracker = list()
        self.labels = list()
        self.points = list()

        if not os.path.isfile(self.vid_file):
            print("[Error] {} not found ".format(self.vid_file))
            raise FileNotFoundError
        try:
            print("[INFO] initializing object detector")
            detector_instance = OBJECT_DETECTOR(cfg.model_path, cfg.model_cfg, cfg.obj_data)
        except ImportError as err:
            raise err
        print("[INFO] starting video stream...")
        vs = cv2.VideoCapture(self.vid_file)
        fps = FPS().start()
        self.count_id = 0
        while True:
            ret, self.frame = vs.read()
            if not ret:
                break
            self.frame = cv2.resize(self.frame, (500, 500), interpolation=cv2.INTER_LINEAR)
            (H, W) = self.frame.shape[:2]
            # self.frame = self.frame[32:720, 250:720]
            # run detector on the frame to get metadata as there is no obj for tracking
            if len(self.tracker) == 0:
                meta_data = detector_instance.darknet_predict(self.frame)
                for label, cnf, cords in meta_data:
                    if label == self.track_obj:
                        self.frame = self.draw_boxes(self.frame, cords, tracking=False)
                        rect = dlib.rectangle(cords[0], cords[1], cords[2], cords[3])
                        t = dlib.correlation_tracker()
                        t.start_track(self.frame, rect)
                        self.tracker.append(t)
                        self.points.append(rect)
            else:
                for k, rect in enumerate(self.points):
                    self.tracker[k].update(self.frame)
                    pos = self.tracker[k].get_position()
                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    self.draw_boxes(self.frame, [startX, startY, endX, endY], tracking=True, objid=self.track_obj)
                    ## un-comment changes if you want to track small or large objects
                    # if endY - startY < 50 or endX - startX < 25:
                    #     print("popping", endY - startY, H)
                    #     self.tracker.pop(k)
                    #     self.points.pop(k)
                    # elif endY > H + 25 or endX > W+5:
                    #     print("popping",endY,H)
                    #     self.tracker.pop(k)
                    #     self.points.pop(k)
                    # else:
                    #     continue
            cv2.imshow("Frame", self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            fps.update()
        fps.stop()
        vs.release()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        # close all windows
        cv2.destroyAllWindows()

    def draw_boxes(self, frame, cords, tracking=False, objid=None):
        xmin, ymin, xmax, ymax = cords
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        if tracking:
            cv2.putText(frame, "Tracking " + str(objid), (xmax, ymax - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
        return frame


if __name__ == '__main__':
    tracker_instance = OBJECT_TRACKING('test4.webm', tracking_obj='person')
