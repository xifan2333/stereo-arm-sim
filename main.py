"""
基于双目视觉三维重建技术的机械臂控制系统
主程序入口

遵循小步骤迭代原则，当前实现：
1. 摄像头打开和显示
2. 视差计算和显示
"""

import cv2
from src.utils.logger import setup_logger, get_logger
from src.vision.camera import StereoCamera
from src.vision.stereo import StereoMatcher


def main():
    """主函数"""
    # 初始化日志系统
    setup_logger()
    logger = get_logger()

    logger.info("=== 基于双目视觉三维重建技术的机械臂控制系统 ===")
    logger.info("初始化中...")

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

            # 计算视差
            try:
                disp_left, disp_right, disp_color = stereo_matcher.compute_disparity(
                    frame_left, frame_right
                )

                # 显示视差图
                stereo_matcher.show_disparity(disp_color, disp_right)

            except Exception as e:
                logger.error(f"视差计算错误: {e}")

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
        logger.success("程序已正常退出")


if __name__ == "__main__":
    main()
