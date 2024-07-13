import os
import time
import datetime

import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np

from src.UNet import UNet
from train_utils import create_lr_scheduler

class SegmentationPresetEval:
    def __init__(self, mean=(0.786, 0.518, 0.784), std=(0.153, 0.210, 0.113)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transforms(img)

def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = 1  # 设置为1，因为要对单张图像进行预测

    # 分割类别数 + 背景
    num_classes = args.num_classes + 1
    crop_size = 480
    mean = (0.787, 0.511, 0.785)
    std = (0.157, 0.213, 0.116)

    # 加载模型
    model = create_model(num_classes=num_classes)
    model = model.to(device)

    # 加载权重
    checkpoint = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # 设置模型为评估模式
    model.eval()

    # 预处理变换
    transform = SegmentationPresetEval(mean=mean, std=std)

    # 读取待分割的图像文件夹
    # img_folder = "IMG/images"
    img_folder="./dataset/DRIVE/test/images"
    img_names = [name for name in os.listdir(img_folder) if name.endswith(".tif")]
    # 创建结果保存文件夹
    result_file="results/result_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(result_file):
        os.makedirs(result_file)

    with torch.no_grad():
        for img_name in img_names:
            # 读取图像
            img_path = os.path.join(img_folder, img_name)
            img = Image.open(img_path).convert('RGB')

            # 图像预处理
            img = transform(img)
            img = img.unsqueeze(0)  # 增加批次维度

            # 将图像移到指定设备上
            img = img.to(device)

            # 进行预测
            pred = model(img)
            pred = torch.argmax(pred['out'], dim=1)  # 获取预测结果

            # 将预测结果保存为图像
            pred_img = pred.squeeze().cpu().numpy()
            pred_img = (pred_img * 255).astype(np.uint8)
            pred_img = Image.fromarray(pred_img)
            save_path = os.path.join(result_file, f"pred_{img_name}")
            pred_img.save(save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch prediction")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="prediction device")
    parser.add_argument("--weights-path", default="save_weights/best_model.pth", help="path to the saved weights")
    args = parser.parse_args()
    main(args)
