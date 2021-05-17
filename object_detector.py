import os
from os.path import join as join_pth
from pathlib import Path

import cv2
import numpy as np

sys_path = Path().absolute()
os.environ["DARKNET_PATH"] = join_pth(sys_path, 'yolov4_darknet')
from yolov4_darknet import darknet as dn

dn.set_gpu(0)


class OBJECT_DETECTOR:
    def __init__(self, model_path: str, yolo_cfg: str, obj_data: str, cnf_thresh=0.9):
        """
        Load darknet model.
        :param model_path: path for yolo model
        :param darknet_path: darknet path with libdarknet.so
        :param Yolo config  file

        :param Obj data path
        :param confidence threshould
        """
        self.model_path = model_path
        self.yolo_cfg = yolo_cfg
        self.obj_data = obj_data
        self.confidence_thesh = cnf_thresh

        self.network, self.cls_names, self.colors = OBJECT_DETECTOR.load_network(self.model_path, self.obj_data,
                                                                                 self.yolo_cfg)
        self.nwidth = dn.network_width(self.network)
        self.nheight = dn.network_height(self.network)

    @staticmethod
    def load_network(model: str, obj: str, cfg: str):
        try:
            network, class_names, colors = dn.load_network(cfg, obj, model)
        except ImportError as err:
            raise err
        return network, class_names, colors

    def convert_to_model_input(self, image: np.ndarray):
        # Darknet doesn't accept numpy images.
        # Create one with image we reuse for each detect

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.nwidth, self.nheight),
                                   interpolation=cv2.INTER_LINEAR)
        return image_resized

    def transform_detections(self, image: np.ndarray, metadata: list):
        img_metadata = list()
        for values in metadata:
            label_name, score, cords = values
            cords = list(map(lambda x: float(x) / self.nwidth, cords))
            x, w = int(cords[0] * image.shape[1]), int(cords[2] * image.shape[1])
            y, h = int(cords[1] * image.shape[0]), int(cords[3] * image.shape[0])
            bbox = [x, y, w, h]
            bbox = dn.bbox2points(bbox)
            img_metadata.append([label_name, score, bbox])
        return img_metadata

    def darknet_predict(self, input_image: np.ndarray):
        converted_image = self.convert_to_model_input(input_image)
        darknet_image = dn.make_image(self.nwidth, self.nheight, 3)
        dn.copy_image_from_bytes(darknet_image, converted_image.tobytes())

        detections = dn.detect_image(self.network, self.cls_names, darknet_image, thresh=self.confidence_thesh)
        dn.free_image(darknet_image)
        metadata = self.transform_detections(input_image, detections)
        # pred_image  = dn.draw_boxes(metadata,input_image,self.colors)
        return metadata


if __name__ == '__main__':
    img_path = join_pth(sys_path, 'yolov4_darknet/data/person.jpg')
    darknet_dir = join_pth(sys_path, 'yolov4_darknet')
    model_path = join_pth(sys_path, "yolov4.weights")
    model_cfg = join_pth(darknet_dir, "cfg/yolov4.cfg")
    obj_data = join_pth(darknet_dir, "cfg/coco.data")
    detector_instance = OBJECT_DETECTOR(model_path, model_cfg, obj_data)
