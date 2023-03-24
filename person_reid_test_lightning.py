from copy import deepcopy
from dataclasses import dataclass

import cv2 as cv
import rospy
import torch
from PIL import Image
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes

from data.transforms import build_transforms
from engine.reid_module import PersonReidModule
from utils import setup_cli, setup_loggers


@dataclass
class InitData:
    init_image: torch.tensor
    init_vector: torch.tensor


def box_callback(msg):
    global flow_boxes
    flow_boxes = msg


def get_box():
    global flow_boxes
    return deepcopy(flow_boxes)


def keep_only_person_boxes(boxes):
    return list(filter(lambda x: x.label == 'person', boxes))


def calc_euclidean(x1, x2):
    return (x1 - x2).pow(2).sum(1)


bridge = CvBridge()
flow_boxes = None


def main(cfg):
    box_topic = '/YD/boxes'
    device = 'cuda:0'

    model = PersonReidModule(cfg, train_num_classes=751, val_num_queries=3368)
    model.load_from_checkpoint(cfg.TEST.WEIGHT)
    model.eval()
    model.to(device)

    transforms = build_transforms(cfg, is_train=False)

    rospy.init_node('test')

    rospy.Subscriber(
        box_topic,
        ObjectBoxes,
        box_callback,
        queue_size=1
    )
    rospy.wait_for_message(box_topic, ObjectBoxes)

    __isCaptured = False
    init_datas = []

    while not rospy.is_shutdown():
        boxes = get_box()
        if boxes is None:
            continue

        frame = bridge.compressed_imgmsg_to_cv2(boxes.source_img)
        if __isCaptured:
            person_boxes = keep_only_person_boxes(boxes.boxes)
            for person_box in person_boxes:
                crop_image = bridge.compressed_imgmsg_to_cv2(person_box.source_img)
                if cfg.DATALOADER.USE_RGB:
                    crop_image = cv.cvtColor(crop_image, cv.COLOR_BGR2RGB)
                crop_image = Image.fromarray(crop_image)
                blob = transforms(crop_image).to(device).unsqueeze(0)
                embedding = model(blob).detach()
                front_dist = calc_euclidean(embedding, init_datas[0].init_vector) / 100
                back_dist = calc_euclidean(embedding, init_datas[1].init_vector) / 100
                front_dist1 = calc_euclidean(embedding, init_datas[2].init_vector) / 100
                back_dist1 = calc_euclidean(embedding, init_datas[3].init_vector) / 100
                threshold = 1.5
                yes = front_dist <= threshold or back_dist <= threshold or front_dist1 <= threshold or back_dist1 <= threshold
                print(front_dist, back_dist, front_dist1, back_dist1)
                if yes:
                    color = (32, 255, 0)
                else:
                    color = (32, 0, 255)

                cv.rectangle(frame, (person_box.x1, person_box.y1), (person_box.x2, person_box.y2), color, 5)

        cv.imshow('test', frame)
        key = cv.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            break

        elif key == ord('c'):
            rx, ry, rw, rh = cv.selectROI('test', frame)
            init_image = frame[ry:ry + rh, rx:rx + rw, :].copy()
            init_image = Image.fromarray(init_image)
            blob = transforms(init_image).to(device).unsqueeze(0)
            init_data = InitData(None, None)
            init_data.init_image = blob
            init_data.init_vector = model(blob).detach()
            init_datas.append(init_data)
            if len(init_datas) >= 4:
                __isCaptured = True


if __name__ == '__main__':
    cfg, args = setup_cli()
    logger = setup_loggers(args, None)
    main(cfg)
