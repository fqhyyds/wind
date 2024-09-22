import argparse #解析命令行参数的库
import json  # 实现字典列表和JSON字符串之间的相互解析
from pathlib import Path  # Path能够更加方便得对字符串路径进行处理
from threading import Thread  # python中处理多线程的库
import os
import numpy as np  # 矩阵计算基础库
import torch  # pytorch 深度学习库
import yaml  # yaml是一种表达高级结构的语言 易读 便于指定模型架构及运行配置
from models.experimental import attempt_load  # 调用models文件夹中的experimental.py文件中的attempt_load函数 目的是加载模型
from tqdm import tqdm  # 用于直观显示进度条的一个库 看起来很舒服
# 以下调用均为utils文件夹中各种已写好的函数
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path
from utils.loss import compute_loss
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target
from utils.torch_utils import select_device, time_synchronized

if __name__ == '__main__':
    """
    以下为 用解析器解析命令行设置test.py相关的参数
    weights:选择测试时的模型权重文件，默认为yolov5s.pt
    data:选择测试时的data配置yaml格式文件，默认为coco128.yaml
    batch-size:前向传播的batch尺寸, 默认32
    img-size:输入图片分辨率大小, 默认640
    conf-thres:筛选框的时候的置信度阈值, 默认0.001
    iou-thres:进行NMS的时候的IOU阈值, 默认0.6
    save-json:是否按照coco的json格式保存预测框，并且使用cocoapi做评估(需要同样coco的json格式的标签), 默认False
    task:设置测试形式, 默认val。
    device:测试的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3
    single-cls:数据集是否只有一个类别
    augment:测试时是否使用TTA(test time augmentation)
    verbose:是否详细记录每一类的mAP指标
    save-txt:是否以txt文件的形式保存模型预测的bounding box
    save-conf:在保存txt文件当中的labels是否包含置信度
    save-json:是否通过cocoapi来保存json结果文件
    project:默认储存测试过程的相对路径，默认为runs/test
    name:默认保存子文件夹的名称，默认为exp，若果exp已经存在会添加递加数字
    exist-ok:已存在的项目/名称可用，不继续增加。
    """
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()  # 解析上述参数
    opt.save_json |= opt.data.endswith('coco.yaml')  # |为或 两者中有一个正确即opt.save_json为正确
    opt.data = check_file(opt.data)  # 检查是否存在相关的配置文件
    print(opt)  # 打印相关的配置
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    if opt.task in ['val', 'test']:  # 在评估和测试任务中对相关参数进行赋值
        test(opt.data,opt.weights,opt.batch_size,opt.img_size,opt.conf_thres,opt.iou_thres,opt.save_json,opt.single_cls,opt.augment,opt.verbose,save_txt=opt.save_txt,save_conf=opt.save_conf,)
    elif opt.task == 'study':  # 在不同的模型下 进行性能测试
        for weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # 保存的文件名
            x = list(range(320, 800, 64))  # x坐标轴
            y = []  # y 坐标
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False)
                y.append(r + t)  # 返回相关结果和时间
            np.savetxt(f, y, fmt='%10.4g')  # 将y输出保存
    os.system('zip -r study.zip study_*.txt')  # 命令行执行命令将study文件进行压缩
    plot_study_txt(f, x)  # 调用plot·s.py中的函数

    print_mutation(hyp.copy(), results, yaml_file, opt.bucket)
