"""
基于双目视觉三维重建技术的机械臂控制系统
主程序入口

遵循小步骤迭代原则，当前实现：
1. 摄像头打开和显示
2. 视差计算和显示
3. YOLO 物体检测和分割
4. 三维重建（提取物体3D信息）
"""

import cv2
import numpy as np
from src.utils.logger import setup_logger, get_logger
from src.utils.config import config
from src.vision.camera import StereoCamera
from src.vision.stereo import StereoMatcher
from src.detection.detector import YOLODetector
from src.reconstruction.pointcloud import extract_object_pointcloud, calculate_object_3d_info
from src.reconstruction import Viewer3D


def main():
    """主函数"""
    # 初始化日志系统
    setup_logger()
    logger = get_logger()

    logger.info("=== 基于双目视觉三维重建技术的机械臂控制系统 ===")
    logger.info("初始化中...")

    # 读取配置
    detection_config = config.get_section("detection")
    show_3d_info = detection_config.get("show_3d_info", True)
    info_bg_color = tuple(detection_config.get("info_bg_color", [255, 255, 255]))
    info_text_color = tuple(detection_config.get("info_text_color", [0, 0, 0]))
    info_font_scale = detection_config.get("info_font_scale", 0.4)
    info_thickness = detection_config.get("info_thickness", 1)

    # 创建摄像头对象
    camera = StereoCamera()

    # 打开摄像头
    if not camera.open():
        logger.error("无法打开摄像头，程序退出")
        return

    # 创建立体匹配器
    logger.info("初始化立体匹配器...")
    try:
        stereo_matcher = StereoMatcher()
    except Exception as e:
        logger.error(f"立体匹配器初始化失败: {e}")
        camera.release()
        return

    # 创建 YOLO 检测器
    logger.info("初始化 YOLO 检测器...")
    try:
        detector = YOLODetector()
    except Exception as e:
        logger.error(f"YOLO 检测器初始化失败: {e}")
        camera.release()
        return

    # 创建 3D 可视化器
    logger.info("初始化 3D 可视化器...")
    try:
        viewer3d = Viewer3D()
        viewer3d.show(block=False)  # 非阻塞模式显示
    except Exception as e:
        logger.error(f"3D 可视化器初始化失败: {e}")
        camera.release()
        return

    logger.info("系统就绪，按 'q' 或 ESC 键退出")

    try:
        while True:
            # 读取帧
            ret, frame_left, frame_right = camera.read_frame()

            if not ret:
                logger.warning("读取帧失败")
                continue

            # 显示左右画面
            camera.show_frames(frame_left, frame_right)

            # 计算视差和3D点云
            try:
                disp_left, disp_right, disp_color, xyz_pointcloud = (
                    stereo_matcher.compute_disparity(frame_left, frame_right)
                )

                # 显示视差图
                stereo_matcher.show_disparity(disp_color, disp_right)

            except Exception as e:
                logger.error(f"视差计算错误: {e}")
                xyz_pointcloud = None

            # YOLO 检测（在左图上）
            try:
                detections = detector.detect(frame_left)

                # 可视化检测结果
                vis_image = detector.visualize(frame_left, detections)

                # 如果有检测结果且点云可用，计算3D信息
                if len(detections) > 0 and xyz_pointcloud is not None:
                    for det in detections:
                        # 提取物体点云
                        try:
                            obj_points = extract_object_pointcloud(
                                xyz_pointcloud, mask=det.mask, bbox=det.bbox
                            )

                            # 调试：打印点云分布
                            if len(obj_points) > 0:
                                logger.info(
                                    f"点云X范围: [{np.min(obj_points[:, 0]):.1f}, {np.max(obj_points[:, 0]):.1f}] mm, "
                                    f"Y范围: [{np.min(obj_points[:, 1]):.1f}, {np.max(obj_points[:, 1]):.1f}] mm, "
                                    f"Z范围: [{np.min(obj_points[:, 2]):.1f}, {np.max(obj_points[:, 2]):.1f}] mm"
                                )

                            # 计算3D中心和尺寸
                            center_mm, dims_mm, confidence = calculate_object_3d_info(
                                obj_points
                            )

                            # 转换为厘米
                            center_cm = center_mm / 10.0
                            dims_cm = dims_mm / 10.0

                            # 调试：打印相机坐标系的尺寸
                            logger.info(
                                f"{det.cls_name} 相机坐标系尺寸(mm): "
                                f"X={dims_mm[0]:.1f}, Y={dims_mm[1]:.1f}, Z={dims_mm[2]:.1f}"
                            )

                            # 在检测图像上显示3D信息
                            if show_3d_info:
                                info_text = (
                                    f"Pos: ({center_cm[0]:.1f}, {center_cm[1]:.1f}, {center_cm[2]:.1f}) cm\n"
                                    f"Size: ({dims_cm[0]:.1f}, {dims_cm[1]:.1f}, {dims_cm[2]:.1f}) cm"
                                )

                                # 在检测框上方显示信息（白底黑字）
                                y_offset = det.y1 - 40
                                font = cv2.FONT_HERSHEY_SIMPLEX

                                for i, line in enumerate(info_text.split("\n")):
                                    # 计算文本尺寸
                                    (text_w, text_h), baseline = cv2.getTextSize(
                                        line, font, info_font_scale, info_thickness
                                    )

                                    # 绘制白色背景
                                    y_pos = y_offset + i * 15
                                    cv2.rectangle(
                                        vis_image,
                                        (det.x1, y_pos - text_h - 2),
                                        (det.x1 + text_w + 4, y_pos + baseline),
                                        info_bg_color,
                                        -1,
                                    )

                                    # 绘制黑色文字
                                    cv2.putText(
                                        vis_image,
                                        line,
                                        (det.x1 + 2, y_pos),
                                        font,
                                        info_font_scale,
                                        info_text_color,
                                        info_thickness,
                                    )

                            # 添加物体到3D场景
                            obj_info = {
                                'center_cm': center_cm,
                                'dims_cm': dims_cm,
                                'points': obj_points / 10.0,  # 转为厘米
                                'class_name': det.cls_name,
                                'confidence': confidence
                            }
                            viewer3d.add_object(obj_info)

                            logger.debug(
                                f"{det.cls_name}: 位置={center_cm}, 尺寸={dims_cm}, 置信度={confidence:.2f}"
                            )

                        except Exception as e:
                            logger.warning(f"物体 {det.cls_name} 3D信息提取失败: {e}")

                # 更新3D显示
                viewer3d.update()

                # 显示检测结果
                detector.show_detections(vis_image)

            except Exception as e:
                logger.error(f"YOLO 检测错误: {e}")

            # 检查退出键
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q'键或ESC键
                logger.info("用户请求退出")
                break

    except KeyboardInterrupt:
        logger.info("用户中断程序")

    except Exception as e:
        logger.error(f"程序运行错误: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 清理资源
        logger.info("清理资源中...")
        camera.release()
        viewer3d.close()
        logger.success("程序已正常退出")


if __name__ == "__main__":
    main()
