"""
draw_proposal -

Author:霍畅
Date:2024/5/27
"""
import os
import sys
import cv2
from mmdet.apis import init_detector
from mmengine import Config
import mmcv
import torch
from mmdet.structures import DetDataSample
import numpy as np
import matplotlib.pyplot as plt
from mmcv.ops import nms

def main(image_path, config_file, checkpoint_file):
    # 检查路径是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path '{image_path}' does not exist.")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file path '{config_file}' does not exist.")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file path '{checkpoint_file}' does not exist.")

    # 初始化模型
    cfg = Config.fromfile(config_file)
    model = init_detector(cfg, checkpoint_file, device='cuda:0')

    # 加载测试图像
    img = mmcv.imread(image_path)

    # 将图像转换为张量并添加批次维度
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to('cuda:0')

    # 准备图像元数据
    img_meta = {
        'ori_shape': (333, 500),
        'img_shape': img.shape[2:],
        'pad_shape': (608, 928),
        'scale_factor': (1.802, 1.8018018018018018),
        'flip': False,
        'flip_direction': None
    }

    # 创建 DetDataSample 实例
    data_sample = DetDataSample()
    data_sample.set_metainfo(img_meta)

    # 自定义前向传播函数以提取RPN的proposals
    def get_rpn_proposals(model, img, data_sample,score_threshold=0.95, nms_threshold=0.2):
        # 提取特征
        x = model.extract_feat(img)
        # 获取RPN输出
        proposals_list = model.rpn_head.predict(x, [data_sample])
        proposals = proposals_list[0].numpy()
        proposals = proposals.cpu().numpy()  # 转换为 numpy 数组
        # 过滤低置信度的proposals
        scores = proposals.scores
        bboxes = proposals.bboxes
        print("Total boxes = ", len(bboxes))
        high_score_indices = scores > score_threshold
        bboxes = bboxes[high_score_indices]
        scores = scores[high_score_indices]
        print("After score filter = ", len(bboxes))
        # 进行非极大值抑制（NMS）
        bboxes = nms(torch.tensor(bboxes), torch.tensor(scores), iou_threshold=nms_threshold)
        print("After NMS = ", len(bboxes[0]))
        return bboxes[0][:, :4]
    # 获取RPN proposals
    with torch.no_grad():
        proposals = get_rpn_proposals(model, img, data_sample)

    # 重新加载原始图像
    original_img = mmcv.imread(image_path)

    # 检查图像类型并转换
    if original_img.dtype != np.uint8:
        original_img = original_img.astype(np.uint8)

    # 提取 proposals 并转换回原始图像坐标
    bboxes = proposals.numpy()

    # 检查图像类型并转换
    if original_img.dtype != np.uint8:
        original_img = original_img.astype(np.uint8)

    # 绘制每个 proposal 框
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.astype(int)
        original_img = cv2.rectangle(original_img, (x1, y1), (x2, y2), (255, 255, 0), 1)

    # 使用matplotlib显示图像
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    # 确保输出目录存在
    output_dir = 'my_code/image_result'
    os.makedirs(output_dir, exist_ok=True)

    # 获取原始图片名称并生成结果图片路径
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_proposals.jpg")

    # 保存图像
    plt.imsave(output_path, cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    import argparse
    # 默认使用faster-rcnn训练的 epoch 9 模型
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