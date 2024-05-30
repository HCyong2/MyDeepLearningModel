"""
test_one_picture -

Author:霍畅
Date:2024/5/24
"""
import sys
import os
import mmcv
import cv2
from mmengine import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules


def main(image_path, config_file, checkpoint_file):
    # 检查路径是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path '{image_path}' does not exist.")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file path '{config_file}' does not exist.")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file path '{checkpoint_file}' does not exist.")

    # 初始化检测模型
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(config_file)
    model = init_detector(cfg, checkpoint_file, device='cuda:0')

    # 测试单张图片
    result = inference_detector(model, image_path)

    # 将结果包装成DetDataSample
    data_sample = DetDataSample()
    data_sample.pred_instances = result.pred_instances

    # 使用可视化工具保存结果图像
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    # 打开图像
    img = mmcv.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 确保输出目录存在
    output_dir = 'my_code/image_result'
    os.makedirs(output_dir, exist_ok=True)

    # 获取原始图片名称并生成结果图片路径
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_result.jpg")

    # 绘制并保存结果
    visualizer.add_datasample(
        name='result',
        image=img_rgb,
        data_sample=data_sample,
        draw_gt=False,
        wait_time=0,
        show=True,  # 设置为True以显示图像
        out_file=output_path  # 保存结果图像
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test a single image with MMDetection')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--config', default=r'my_code/faster-rcnn.py',
                        help='Path to the config file')
    parser.add_argument('--checkpoint', default=r'my_code\my_model\faster-rcnn\epoch_9.pth',
                        help='Path to the checkpoint file')

    args = parser.parse_args()

    try:
        main(args.image_path, args.config, args.checkpoint)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)