"""
基于双目视觉三维重建技术的机械臂控制系统
主程序入口

遵循小步骤迭代原则，当前实现摄像头打开和显示功能
"""

import cv2
from src.utils.logger import setup_logger, get_logger
from src.vision.camera import StereoCamera


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

    logger.info("摄像头已就绪，按 'q' 或 ESC 键退出")

    try:
        while True:
            # 读取帧
            ret, frame_left, frame_right = camera.read_frame()

            if not ret:
                logger.warning("读取帧失败")
                continue

            # 显示左右画面
            camera.show_frames(frame_left, frame_right)

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
