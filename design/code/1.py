import json  # 实现字典列表和JSON字符串之间的相互解析
from pathlib import Path  # Path能够更加方便得对字符串路径进行处理
from threading import Thread  # python中处理多线程的库

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


@torch.no_grad()
# 测试函数 输入为测试过程中需要的各种参数
def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.5,  # 为NMS设置的iou阈值
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # 存储测试图片的路径
         save_txt=False,  # 自动实现对测试图片的标注
         save_conf=False,  # 保存置信度
         plots=True,
         log_imgs=0):  # 已记录图片的数量

    # 初始化/加载模型 并设置设备
    training = model is not None  # 有模型则 training 为True
    if training:  # 调用train.py
        device = next(model.parameters()).device  # 获得记录在模型中的设备 next为迭代器

    else:
        set_logging()  # 调用general.py文件中的函数 设置日志 opt对象main中解析传入变量的对象
        device = select_device(opt.device, batch_size=batch_size)  # 调用torch_utils中select_device来选择执行程序时的设备
        save_txt = opt.save_txt  # 获取保存测试之后的label文件路径 格式为txt

        # 路径
        # 调用genera.py中的increment_path函数来设置保存文件的路径
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        # mkdir创建路径最后一级目录
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

        # 加载模型
        # 加载模型为32位浮点数模型（权重参数） 调用experimental.py文件中的attempt_load函数
        model = attempt_load(weights, map_location=device)
        # 调用general.py中的check_img_size函数来检查图像分辨率能否被32整除
        imgsz = check_img_size(imgsz, s=model.stride.max())
        # 精度减半
    # 如果设备类型不是cpu 则将模型由32位浮点数转换为16位浮点数
    half = device.type != 'cpu'
    if half:
        model.half()

    # 加载配置
    model.eval()  # 将模型转换为测试模式 固定住dropout层和Batch Normalization层
    is_coco = data.endswith('coco.yaml')  # 判断输入的数据yaml文件是否是coco.yaml文件
    with open(data) as f:  # 打开data（yaml格式）文件
        data = yaml.load(f, Loader=yaml.FullLoader)  # 获取模型配置的字典格式文件
    check_dataset(data)  # 调用general.py中的check_dataset函数来检查数据文件是否正常
    nc = 1 if single_cls else int(data['nc'])  # 确定检测的类别数目
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # mAP@0.5:0.95 的iou向量
    niou = iouv.numel()  # numel为pytorch预置函数 用来获取张量中的元素个数

    # 日志
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # 权重和偏置 wandb为可视化权重和各种指标的库
    except ImportError:
        log_imgs = 0

        # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # 创建一张全为0的图片（四维张量）
        # 利用空图片对模型进行测试 （只在运行设备不是cpu时进行）
        _ = model(img.half() if half else img) if device.type != 'cpu' else None
        path = data['test'] if opt.task == 'test' else data['val']  # 如果任务为test 则获得yaml文件中测试的路径
        # 调用datasets.py文件中的create_dataloader函数创建dataloader
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True)[0]

    seen = 0  # 初始化已完成测试的图片数量
    confusion_matrix = ConfusionMatrix(nc=nc)  # 调用matrics中函数 存储混淆矩阵
    # 获取模型训练中存储的类别名字数据
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()  # 调用general.py中的函数 来转换coco的类
    # 为后续设置基于tqdm的进度条作基础
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    # 初始化detection中各个指标的值 t0和t1为时间
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    # 初始化网络训练的loss
    loss = torch.zeros(3, device=device)
    # 初始化json文件涉及到的字典、统计信息、AP、每一个类别的AP、图片汇总
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        # 将图片数据拷贝到device（GPU）上面
        img = img.to(device, non_blocking=True)
        # 将图片从64位精度转换为32位精度
        img = img.half() if half else img.float()
        # 将图像像素值0-255的范围归一化到0-1的范围
        img /= 255.0
        # 对targets也做同样拷贝的操作
        targets = targets.to(device)
        # 四个变量分别代表batchsize、通道数目、图像高度、图像宽度
        nb, _, height, width = img.shape
        with torch.no_grad():  # 强制之后进行的过程不生成计算图 开始运行模型

    t = time_synchronized()  # 调用torch_utils中的函数 开始计时
    inf_out, train_out = model(img, augment=augment)  # 输入图片进行模型推断 返回推断结果及训练结果
    t0 += time_synchronized() - t  # t0 为累计的各个推断所用时间

    # 计算损失
    if training:  # 如果在训练时进行test
        # loss 包含bounding box 回归的GIoU、object和class 三者的损失
        loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]

        # 运行NMS 目标检测的后处理模块 用于删除冗余的bounding box
    targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # 对targets进行处理并拷贝到GPU
    # 提取bach中每一张图片的目标的label
    lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_txt else []

    t = time_synchronized()  # 计算NMS过程所需要的时间
    # 调用general.py中的函数 进行非极大值抑制操作
    output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb)
    t1 += time_synchronized() - t


# Statistics per image
# si代表第si张图片，pred是对应图片预测的label信息
for si, pred in enumerate(output):
    labels = targets[targets[:, 0] == si, 1:]  # 调取第si张图片的label信息
    nl = len(labels)  # nl为图片检测到的目标个数
    tcls = labels[:, 0].tolist() if nl else []  # 检测到的目标的类别 label矩阵的第一列
    path = Path(paths[si])  # 找到第si张照片对应的文件路径
    seen += 1  # 处理的图片增加1

    if len(pred) == 0:  # 如果没有预测到目标则
        if nl:  # 同时有label信息
            # stats初始化为一个空列表[] 此处添加一个空信息
            # 添加的每一个元素均为tuple 其中第二第三个变量为一个空的tensor
            stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
        continue
    # Predictions
    if single_cls:
        pred[:, 5] = 0
        # 预测
    predn = pred.clone()  # 对pred进行深复制
    # 调用general.py中的函数 将图片调整为原图大小
    scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

    # 将结果添加到文本文档里面
    if save_txt:
        # shapes具体变量设置应看dataloader 此处应为提取长和宽并构建新tensor gn
        # 对torch.tensor()[[1,0]]操作可构建一个新tensor其中第一行为内层列表中第一个索引对应的行
        gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            """
            将xyxy格式的坐标转换成xywh坐标 调用general.py中的函数
            xyxy格式为记录bounding box 的左上角和右下角坐标
            xywh格式为记录中心点坐标和bounding box的宽和高
            """
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            # line为按照YOLO格式输出的测试结果 [类别 x y w h]
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
            # 将上述test得到的信息输出保存 输出为xywh格式 coco数据格式也为xywh格式
            with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

    # 记录图片测试结果相关信息 并 存储在日志当中
    if plots and len(wandb_images) < log_imgs:
        # 一个 包含嵌套字典的列表的数据结构 存储 一个box对应的数据信息
        box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                     "class_id": int(cls),
                     "box_caption": "%s %.3f" % (names[cls], conf),
                     "scores": {"class_score": conf},
                     "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
        boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
        # 记录每一张图片 每一个box的相关信息 wandb_images 初始化为一个空列表
        wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

    # 将信息添加到JSON字典当中
    if save_json:
        # 储存的格式 [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
        # 记录的信息有id box xy to top-left 得分等 如下所示
        image_id = int(path.stem) if path.stem.isnumeric() else path.stem
        box = xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(pred.tolist(), box.tolist()):
            jdict.append({'image_id': image_id,
                          'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                          'bbox': [round(x, 3) for x in b],
                          'score': round(p[4], 5)})

    # 初始化时将所有的预测都当做错误
    # niou为iou阈值的个数
    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
    if nl:  # 当检测到的个数不为0时 nl为图片检测到的目标个数
        detected = []  # 目标相关指标
        tcls_tensor = labels[:, 0]  # 获得类别的tensor

        # 目标的boundingbox
        tbox = xywh2xyxy(labels[:, 1:5])  # xywh格式转换为xyxy格式
        # 按照原本图片的缩放标准 对bounding box 进行缩放
        scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
        if plots:  # 这里是绘制混淆矩阵 见metrics.py 代码解读
            confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

        # 对于每一个目标类
        for cls in torch.unique(tcls_tensor):
            # numpy.nonzero()函数为获得非零元素的所以 返回值为长为numpy.dim长度的tuple
            # 因此ti为标签box对应的索引 ti = test indices
            ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
            # 因此pi为预测box对应的索引 pi = prediction indices
            pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

            #  对于每个单独的类别寻找检测结果
            if pi.shape[0]:
                # Prediction to target ious
                # 调用general.py中的box_iou函数 返回最大的iou对应对象及相关指标
                ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                # 将检测到目标统一添加到 detected_set集合当中
                detected_set = set()
                for j in (ious > iouv[0]).nonzero(as_tuple=False):
                    d = ti[i[j]]  # 检测到的目标
                    if d.item() not in detected_set:
                        # 将不在检测集合中的目标添加到集合里面
                        detected_set.add(d.item())
                        detected.append(d)
                        # 只有iou大于阈值的才会被认为是正确的目标
                        correct[pi[j]] = ious[j] > iouv  # iou_thres 是一个 1xn 的向量
                        if len(detected) == nl:  # 图像中所有的目标都已经被处理完
                            break

    # 向stats（list）中添加统计指标 格式为：(correct, conf, pcls, tcls)
    stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Plot images
# 画出前三个图片的ground truth和 对应的预测框 Q：当测试时并没有ground truth支持时该如何处理？
if plots and batch_i < 3:
    f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
    # Thread()函数为创建一个新的线程来执行这个函数 函数为plots.py中的plot_images函数
    Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
    f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
    Thread(target=plot_images, args=(img, output_to_target(output), paths, f, names), daemon=True).start()

# 计算上述测试过程中的各种性能指标
stats = [np.concatenate(x, 0) for x in zip(*stats)]  # 转换为对应格式numpy
# 以下性能指标的计算方法及原理 可移步笔者 metrics.py源代码解读博客
if len(stats) and stats[0].any():
    p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
    p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
else:
    nt = torch.zeros(1)

# 按照以下格式来打印测试过程的指标
pf = '%20s' + '%12.3g' * 6  # print format
print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

# 打印每一个类别对应的性能指标
if verbose and nc > 1 and len(stats):
    for i, c in enumerate(ap_class):
        print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

# 打印 推断/NMS过程/总过程 的在每一个batch上面的时间消耗
t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
if not training:
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

# 绘图 绘制混淆矩阵 wandb日志文件中也对测试图片进行可视化（具体可视化样子目前暂不清）
if plots:
    confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    if wandb and wandb.run:
        wandb.log({"Images": wandb_images})
        wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]})

# 保存之前json格式的预测结果，并利用coco的api进行评估
# 因为COCO测试集的标签是给出的 因此此评估过程结合了测试集标签
# 在更多的目标检测场合下 为保证公正测试集标签不会给出 因此以下过程应进行修改
if save_json and len(jdict):
    w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
    anno_json = '../coco/annotations/instances_val2017.json'  # 注释的json格式
    pred_json = str(save_dir / f"{w}_predictions.json")  # 预测的json格式
    print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
    with open(pred_json, 'w') as f:
        json.dump(jdict, f)

    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        # 以下过程为利用官方coco工具进行结果的评测
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        anno = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        eval = COCOeval(anno, pred, 'bbox')
        if is_coco:
            eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    except Exception as e:
        print(f'pycocotools unable to run: {e}')

# 返回结果
if not training:  # 如果不是训练过程则将结果保存到对应的路径
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    print(f"Results saved to {save_dir}{s}")
model.float()  # 将模型转换为适用于训练的状态
maps = np.zeros(nc) + map
for i, c in enumerate(ap_class):
    maps[c] = ap[i]
# 返回对应的测试结果
return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t
