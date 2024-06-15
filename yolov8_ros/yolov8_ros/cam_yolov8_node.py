import time
import threading
import cv2
from typing import List, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints

from sensor_msgs.msg import Image
from yolov8_msgs.msg import Point2D
from yolov8_msgs.msg import BoundingBox2D
from yolov8_msgs.msg import Mask
from yolov8_msgs.msg import KeyPoint2D
from yolov8_msgs.msg import KeyPoint2DArray
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray
from std_srvs.srv import SetBool


class Yolov8Node(Node):

    def __init__(self) -> None:
        super().__init__("cam_yolov8_node")

        # params
        self.declare_parameter("model", "yolov8m.pt")
        model = self.get_parameter("model").get_parameter_value().string_value

        self.declare_parameter("device", "cuda:0")
        self.device = self.get_parameter("device").get_parameter_value().string_value

        self.declare_parameter("threshold", 0.5)
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value

        self.declare_parameter("enable", True)
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value

        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        image_qos_profile = QoSProfile(
            reliability=self.get_parameter("image_reliability").get_parameter_value().integer_value,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.spin_time = time.time()
        self.yolo_time = time.time()

        self.cv_bridge = CvBridge()
        self.yolo = YOLO(model)
        self.yolo.fuse()

        self.frame = None
        self.frame_lock = threading.Lock()
        self.enable_image_publish = False

        # publishers
        self.detection_pub = self.create_publisher(DetectionArray, "detections", 10)
        self.image_pub = self.create_publisher(Image, "image_raw", image_qos_profile)

        # services
        self._srv = self.create_service(SetBool, "enable", self.enable_cb)

        self.get_logger().info("YOLO cam node started")

        # Start the image capture thread
        self.image_capture_thread = threading.Thread(target=self.image_capture)
        self.image_capture_thread.start()

        # Start the YOLO processing thread
        self.yolo_thread = threading.Thread(target=self.yolo_process)
        self.yolo_thread.start()

    def enable_cb(self, req: SetBool.Request, res: SetBool.Response) -> SetBool.Response:
        self.enable = req.data
        res.success = True
        return res

    def parse_hypothesis(self, results: Results) -> List[Dict]:
        hypothesis_list = []
        box_data: Boxes
        for box_data in results.boxes:
            hypothesis = {
                "class_id": int(box_data.cls),
                "class_name": self.yolo.names[int(box_data.cls)],
                "score": float(box_data.conf)
            }
            hypothesis_list.append(hypothesis)
        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        boxes_list = []
        box_data: Boxes
        for box_data in results.boxes:
            msg = BoundingBox2D()
            box = box_data.xywh[0]
            msg.center.position.x = float(box[0])
            msg.center.position.y = float(box[1])
            msg.size.x = float(box[2])
            msg.size.y = float(box[3])
            boxes_list.append(msg)
        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:
        masks_list = []
        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p
        mask: Masks
        for mask in results.masks:
            msg = Mask()
            msg.data = [create_point2d(float(ele[0]), float(ele[1])) for ele in mask.xy[0].tolist()]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]
            masks_list.append(msg)
        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:
        keypoints_list = []
        points: Keypoints
        for points in results.keypoints:
            msg_array = KeyPoint2DArray()
            if points.conf is None:
                continue
            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):
                if conf >= self.threshold:
                    msg = KeyPoint2D()
                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)
                    msg_array.data.append(msg)
            keypoints_list.append(msg_array)
        return keypoints_list

    def yolo_process(self):
        while rclpy.ok():
            if not self.enable:
                time.sleep(0.1)
                continue

            with self.frame_lock:
                if self.frame is None:
                    continue
                frame = self.frame.copy()

            self.yolo_time = time.time()
            results = self.yolo.predict(
                source=frame,
                verbose=False,
                stream=False,
                show=False,
                conf=self.threshold,
                device=self.device,
                classes=[0, 56, 26, 67]
            )
            results: Results = results[0].cpu()
            yolo_predict = time.time()

            if results.boxes:
                hypothesis = self.parse_hypothesis(results)
                boxes = self.parse_boxes(results)

            if results.masks:
                masks = self.parse_masks(results)

            if results.keypoints:
                keypoints = self.parse_keypoints(results)

            # create detection msgs
            detections_msg = DetectionArray()

            for i in range(len(results)):
                aux_msg = Detection()
                if results.boxes:
                    aux_msg.class_id = hypothesis[i]["class_id"]
                    aux_msg.class_name = hypothesis[i]["class_name"]
                    aux_msg.score = hypothesis[i]["score"]
                    aux_msg.bbox = boxes[i]
                if results.masks:
                    aux_msg.mask = masks[i]
                if results.keypoints:
                    aux_msg.keypoints = keypoints[i]
                detections_msg.detections.append(aux_msg)

            # publish detections
            if self.enable_image_publish:
                detections_msg.header.stamp = self.get_clock().now().to_msg()
                self.detection_pub.publish(detections_msg)

            self.get_logger().info(
                f"Full spin_time: {(time.time() - self.spin_time):.3f}S - "
                f"yolo_predict: {(yolo_predict - self.yolo_time):.3f}S")
            self.spin_time = time.time()

    def image_capture(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.get_logger().error("Could not open webcam.")
            return

        while rclpy.ok():
            ret, frame = cap.read()
            if not ret:
                self.get_logger().error("Failed to capture image from webcam.")
                continue

            with self.frame_lock:
                self.frame = frame

            # Convert OpenCV image to ROS Image message
            image_msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            image_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(image_msg)

        cap.release()


def main():
    rclpy.init()
    node = Yolov8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
