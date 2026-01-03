# -*- coding: utf-8 -*-
import threading
import cv2
import matplotlib.pyplot as plt
import sys

# Import modules from our new structure
from src.config import CAMERA_CONFIG, TOTAL_FRAMES
from src.utils.helpers import logger, get_object_id
from src.vision.stereo import StereoProcessor
from src.vision.detection import YOLODetector
from src.vision.tracking import SimpleTracker, PositionTracker, PositionStabilizer
from src.geometry.pointcloud import extract_object_pointcloud
from src.geometry.estimation import calculate_object_depth, calculate_3d_bounding_box_from_points
from src.visualization.plotter import Plotter

# Global variables (now managed or sourced from config)
# These were globals in original main.py, now they should be class attributes or managed differently.
# For demo purposes, keeping some simple ones here or deriving them from config.
planning_result = None  # 规划结果
animation_running = False  # 动画运行状态
detected_objects = []  # 存储检测到的物体
detected_objects_lock = threading.Lock()  # 用于同步检测到的物体列表
selected_object_id = None  # 选中的物体ID
object_bbox_points = {}  # 存储物体的边界框点
stabilization_complete = False  # 稳定检测完成标志
stabilization_counter = 0  # 稳定计数器

last_left_frame_bgr = None
mono_depth_map_mm = None # Still a placeholder for mono depth if implemented
last_mono_frame_idx = 0
object_bbox_rects = {}  # 存储物体的边界框矩形
object_3d_bboxes = {}  # 存储物体的3D包围盒

# --- Initialization ---
stereo_processor = StereoProcessor()
yolo_detector = YOLODetector()
object_tracker = SimpleTracker()
position_tracker = PositionTracker(yolo_model_names=yolo_detector.model.names if yolo_detector.model else {})
stabilizer = PositionStabilizer(window_size=5, ema_alpha=None) # EMA alpha could be from config
plotter = Plotter()

xyz_lock = threading.Lock() # Global lock if multiple threads access xyz data

def main_loop():
    global animation_running, selected_object_id, stabilization_complete, stabilization_counter
    global last_left_frame_bgr, mono_depth_map_mm, last_mono_frame_idx

    # Placeholder for camera capture (e.g., using OpenCV VideoCapture)
    # In a real system, you'd initialize your stereo cameras here.
    cap_l = cv2.VideoCapture(0) # Left camera, adjust index as needed
    cap_r = cv2.VideoCapture(1) # Right camera, adjust index as needed

    if not cap_l.isOpened() or not cap_r.isOpened():
        logger.error("无法打开立体摄像头。请检查设备连接或摄像头索引。")
        sys.exit(1)

    frame_idx = 0
    animation_running = True

    try:
        while animation_running:
            frame_idx += 1

            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()

            if not ret_l or not ret_r:
                logger.error("无法获取图像帧。")
                break

            # 1. 图像校正 (Stereo Rectification)
            # Apply maps generated during StereoProcessor initialization
            rectify_frame_l = cv2.remap(frame_l, stereo_processor.mapLx, stereo_processor.mapLy, cv2.INTER_LINEAR)
            rectify_frame_r = cv2.remap(frame_r, stereo_processor.mapRx, stereo_processor.mapRy, cv2.INTER_LINEAR)

            # Convert to grayscale for stereo matching
            gray_l = cv2.cvtColor(rectify_frame_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(rectify_frame_r, cv2.COLOR_BGR2GRAY)

            # 2. 立体匹配与点云生成 (Stereo Matching & Point Cloud Generation)
            disp_color_l, xyz_l_mm, conf_l, disp_l, disp_r, xyz_r_mm = \
                stereo_processor.stereo_match(gray_l, gray_r)

            # 3. 物体检测 (Object Detection)
            yolo_detections = yolo_detector.detect(rectify_frame_l)
            tracked_detections = object_tracker.update(yolo_detections)

            total_drawn_points = 0

            with detected_objects_lock:
                global detected_objects
                detected_objects = [] # Clear for current frame

                for det in tracked_detections:
                    bbox_2d = det['bbox']
                    class_name = det['class_name']
                    track_id = det.get('track_id') # Get track_id from SimpleTracker
                    obj_id = get_object_id({'track_id': track_id, 'class_name': class_name, 'bbox': bbox_2d})

                    # Extract object point cloud
                    obj_pointcloud_cm, valid_ratio = extract_object_pointcloud(
                        xyz_l_mm / 10.0, bbox_2d, class_name=class_name
                    ) # Convert mm to cm for estimation functions

                    # Estimate 3D position and dimensions
                    obj_center_cm, obj_dims_cm, depth_conf = calculate_object_depth(
                        obj_pointcloud_cm, bbox_2d=bbox_2d,
                        image_shape=CAMERA_CONFIG.image_size, class_name=class_name
                    )

                    if obj_center_cm is not None:
                        # Smooth position
                        smoothed_pos_cm = stabilizer.update(obj_id, obj_center_cm)

                        # Calculate 3D bounding box
                        bbox_3d_info = calculate_3d_bounding_box_from_points(
                            obj_pointcloud_cm, estimated_dimensions=obj_dims_cm, center=smoothed_pos_cm
                        )

                        # Store and update position tracker
                        position_tracker.update_position(obj_id, smoothed_pos_cm, depth_conf, det['class_id'])

                        # Prepare object info for visualization
                        obj_info = {
                            'id': obj_id,
                            'class_name': class_name,
                            'position': smoothed_pos_cm, # In cm
                            'dimensions': obj_dims_cm,   # In cm
                            'confidence': depth_conf,
                            'pointcloud': obj_pointcloud_cm, # In cm
                            'bbox_3d': bbox_3d_info
                        }
                        detected_objects.append(obj_info)
                        if obj_pointcloud_cm is not None:
                            total_drawn_points += obj_pointcloud_cm.shape[0]

            # 4. 状态更新与可视化 (State Update & Visualization)
            plotter.update_display(
                frame_idx, TOTAL_FRAMES, rectify_frame_l, disp_color_l, detected_objects, total_drawn_points
            )

            # Check for keyboard input to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                animation_running = False

    except KeyboardInterrupt:
        logger.info("用户中断程序。")
    except Exception as e:
        logger.error(f"发生错误: {e}")
    finally:
        logger.info("清理资源中...")
        cap_l.release()
        cap_r.release()
        cv2.destroyAllWindows()
        plt.close(plotter.fig)
        logger.info("清理完成。")


if __name__ == "__main__":
    logger.info("启动立体视觉机械臂避障系统...")
    main_loop()