#使用原模型加修改后的损失函数
import torch
import time  # 用于计时
import os.path
import os
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from Dataset_2 import MyDataset  # 假设你的dataset脚本是MyDataset.py
from NewModel_19 import  FusionNetVGG
from tqdm import tqdm
from metric import Evaluator  # 导入Evaluator类
from set_seed import set_seed  # 导入设置随机种子的模块
from CosineAnnealingLR import CosineAnnealingWarmupRestarts
from loss_fn21_1 import  DynamicMiningLoss
# 设置随机种

set_seed(1)  # 设置随机种子为42，这个数字可以是任意的，但是要保证模型可复现的情况下是就需要在相同环境使用相同种子
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 新增环境变量设置
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 修改后的设备定义
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 定义训练设备
# 配置部分 - 方便调参
config = {
    'train_image_dir': 'data/1-train/1-images',  # 修改输入数据路径
    'train_label_dir': 'data/1-train/2-labels',
    'test_image_dir': 'data/2-test/1-images',
    'test_label_dir': 'data/2-test/2-labels',
    'batch_size': 10,  # batch_size大小
    'n_channels': 5,  # n_channels表示输入数据通道数
    'num_classes': 2,  # num_classes表示分的类别
    'learning_rate': 0.001,  # lr
    'num_epochs': 600,  # 运行轮数
    'save_interval': 150,  # 保存模型间隔（间隔20轮）
    'log_dir': 'train_result(NewModel_19_new_086)',  # 运行日志保存文件夹
    'train_log_filename': 'train_result(NewModel_19_new_086)/train_log.txt',  # 运行日志名称。
    'test_log_filename': 'train_result(NewModel_19_new_086)/test_log.txt',  # 运行日志名称
    'best_model_filename': 'train_result(NewModel_19_new_086)/best_model.pth',  # 保存最佳模型的路径}
    # 新增的学习率调度器配置
    'first_cycle_steps': 100,  # 第一次周期的步数
    'cycle_mult': 1.5,  # 每个周期的倍增因子
    'max_lr': 0.001,  # 最大学习率
    'min_lr': 1e-5,  # 最小学习率
    'warmup_steps': 20,  # 热身步数
    'gamma': 0.98,
    'use_augmentation': True}  # 新增数据增强开关} # 学习率衰减因子}} # 学习率衰减因子

# 把数据集加载到MyDataset
train_dataset = MyDataset(config['train_image_dir'], config['train_label_dir'], transform=config['use_augmentation'])
test_dataset = MyDataset(config['test_image_dir'], config['test_label_dir'],transform=False )

# 查看数据集大小;
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print(f'训练数据集的长度为：{train_dataset_size}')
print(f'测试数据集的长度为：{test_dataset_size}')

# 利用DataLoader加载数据
# train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
# 修改DataLoader部分，启用多进程和内存锁页
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=8,          # 根据CPU核心数调整（建议4~8）
    pin_memory=True,         # 加速数据到GPU的传输
    persistent_workers=True  # 避免重复初始化进程
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

# 导入模型并迁移到设备
model = FusionNetVGG(num_classes=2).to(device)
# 创建损失函数并迁移到设备
# 创建损失函数并迁移到设备
loss_fn = DynamicMiningLoss(
    alpha=0.86,    # 交叉熵占比87%
    power=0.5     # 中等强度权重调整
).to(device)

# 创建优化器
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
# scaler = GradScaler()#混合精度训练
# 创建学习率调度器
scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=config['first_cycle_steps'],
                                          cycle_mult=config['cycle_mult'],
                                          max_lr=config['max_lr'],
                                          min_lr=config['min_lr'],
                                          warmup_steps=config['warmup_steps'],
                                          gamma=config['gamma'])

# 创建保存模型的文件夹（确保只创建一次）
if not os.path.exists(config['log_dir']):
    os.makedirs(config['log_dir'])

# 创建Evaluator对象（精度评估器）
evaluator_train = Evaluator(num_class=config['num_classes'])  # 训练集评估器
evaluator_test = Evaluator(num_class=config['num_classes'])  # 测试集评估器

# 打开日志文件，准备写入
train_log_file = open(config['train_log_filename'], 'w')
test_log_file = open(config['test_log_filename'], 'w')

train_log_file.write("Epoch\t"
                     "Train_Loss\t"
                     "Train_IoU_Class_1\t"
                     "Train_IoU_Class_2\t"
                     "Train_Precision_Class_1\t"
                     "Train_Precision_Class_2\t"
                     "Train_Recall_Class_1\t"
                     "Train_Recall_Class_2\t"
                     "Train_F1_Score_Class_1\t"  # 排土场（类别1）
                     "Train_F1_Score_Class_2\t"  # 矿区（类别2）
                     "Train_Dice_Class_1\t"
                     "Train_Dice_Class_2\t"
                     "train_Kappa\t"
                     "Train_OA\t"
                     "Train_FWIoU\n" )

test_log_file.write("Epoch\t"
                    "test_Loss\t"
                    "test_IoU_Class_1\t"
                    "test_IoU_Class_2\t"
                    "test_Precision_Class_1\t"
                    "test_Precision_Class_2\t"
                    "test_Recall_Class_1\t"
                    "test_Recall_Class_2\t"
                    "test_F1_Score_Class_1\t"  # 排土场（类别1）
                    "test_F1_Score_Class_2\t"  # 矿区（类别2）
                    "test_Dice_Class_1\t"
                    "test_Dice_Class_2\t"
                    "test_Kappa\t"
                    "test_OA\t"
                    "test_FWIoU\n")

# 初始化一个变量用于记录当前最佳的测试集IoU第二个值
best_test_iou1 = -1.0  # 设定一个非常小的初始值，以便在第一次测试时能够更新


# 记录整个训练过程的总开始时间
total_start_time = time.time()
total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试的次数

# 初始化变量，保存最佳模型
best_epoch = 0  # 最好精度所在的epoch

# 训练和测试过程
for epoch in range(config['num_epochs']):
    model.train()  # 设置模型进入训练状态
    epoch_start_time = time.time()  # 记录每轮的训练开始时间
    total_train_loss = 0  # 记录这一轮的总损失
    total_batches = 0  # 统计batch数

    # 用tqdm包裹train_dataloader来显示进度条
    pbar = tqdm(train_dataloader, desc=f"训练第{epoch + 1}轮", ncols=100)  # 每轮重新实例化 tqdm
    for data in pbar:
        (input_branch_1, input_branch_2),labels = data
        input_branch_1, input_branch_2, labels =input_branch_1.to(device), input_branch_2.to(device), labels.to(device)

        outputs = model(input_branch_1, input_branch_2)# 把数据导入模型进行训练
        loss = loss_fn(outputs, labels)  # 计算损失值

        # 优化器优化模型
        optimizer.zero_grad()  # 利用优化器进行梯度清零
        loss.backward()  # 利用得到的损失函数，调用反向传播，得到每个参数节点的梯度
        optimizer.step()  # 调用优化器对参数进行优化；至此一次训练结束
        total_train_step += 1  # 记录训练的次数加1

        # 计算准确率
        total_train_loss += loss.item()  # 累加损失
        total_batches += 1  # 增加已处理的batch数
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        pbar.set_postfix(loss=total_train_loss / total_batches, lr=current_lr)
        # 将预测结果与真实标签传入评估器计算精度
        pred_train = outputs.argmax(dim=1).cpu().numpy()
        gt_train = labels.cpu().numpy()
        evaluator_train.add_batch(gt_train, pred_train)  # 使用训练集评估器

    # 计算这一轮的平均损失
    avg_train_loss = total_train_loss / total_batches
    # 打印每一轮的训练时间
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"第{epoch + 1}轮训练完成，训练时间：{epoch_duration:.2f}秒")
    print(f"第{epoch + 1}轮训练的平均损失：{avg_train_loss:.4f}")
    print(f"第{epoch + 1}轮训练的学习率为：{current_lr:.5f}")
    # 计算并打印训练集评估指标
    train_precision = evaluator_train.Precision()
    train_recall = evaluator_train.Recall()
    train_f1_score = evaluator_train.F1()
    train_overall_accuracy = evaluator_train.OA()
    train_iou = evaluator_train.Intersection_over_Union()
    train_dice = evaluator_train.Dice()
    train_pixel_accuracy_class = evaluator_train.Pixel_Accuracy_Class()
    train_fw_iou = evaluator_train.Frequency_Weighted_Intersection_over_Union()
    train_Kappa =  evaluator_train.Kappa()
    print(f"训练集评估指标（第{epoch + 1}轮）：",  f"  IoU: {train_iou}",f"  Precision: {train_precision}", f"  Recall: {train_recall}",
          f"  F1 Score: {train_f1_score}", f"  Dice: {train_dice}",f" Kappa: {train_Kappa:.4f}",
          f"  Overall Accuracy: {train_overall_accuracy:.4f}", f"  Frequency Weighted IoU: {train_fw_iou:.4f}")

    # 在每个epoch结束后更新学习率
    scheduler.step(epoch)
    # 测试步骤开始
    model.eval()  # 设置模型进入验证状态
    total_test_loss = 0  # 记录在整个数据集中的损失

    with torch.no_grad():  # 在with里边的代码就没有了梯度，能够保证不会对其进行调优
        for data in test_dataloader:
            (input_branch_1, input_branch_2), labels = data
            input_branch_1, input_branch_2, labels = input_branch_1.to(device), input_branch_2.to(device), labels.to(
                device)
            outputs = model(input_branch_1, input_branch_2)  # 把数据导入模型进行训练
            loss = loss_fn(outputs, labels)  # 计算测试集的损失
            total_test_loss += loss.item()  # 累加损失

            # 将预测结果与真实标签传入评估器计算精度
            pred_test = outputs.argmax(dim=1).cpu().numpy()
            gt_test = labels.cpu().numpy()
            evaluator_test.add_batch(gt_test, pred_test)
    # 计算测试集的平均损失
    avg_test_loss = total_test_loss / len(test_dataloader)
    print(f"第{epoch + 1}轮测试完成，平均损失为：{avg_test_loss:.2f}")
    # 计算并打印测试集评估指标
    test_precision = evaluator_test.Precision()
    test_recall = evaluator_test.Recall()
    test_f1_score = evaluator_test.F1()
    test_overall_accuracy = evaluator_test.OA()
    test_iou = evaluator_test.Intersection_over_Union()
    test_dice = evaluator_test.Dice()
    test_pixel_accuracy_class = evaluator_test.Pixel_Accuracy_Class()
    test_fw_iou = evaluator_test.Frequency_Weighted_Intersection_over_Union()
    test_Kappa = evaluator_test.Kappa()
    print(f"测试集评估指标（第{epoch + 1}轮）：", f"  IoU: {test_iou}",f"  Precision: {test_precision}",
          f"  Recall: {test_recall}", f"  F1 Score: {test_f1_score}", f"  Dice: {test_dice}",
          f"  Kappa: {test_Kappa:.4f}",f"  Overall Accuracy: {test_overall_accuracy:.4f}",
          f"  Frequency Weighted IoU: {test_fw_iou:.4f}")


    # 保存最佳模型
    # 比较测试集的IoU第二个值（test_iou[1]）是否优于当前最优值
    if test_iou[1] > best_test_iou1:
        best_test_iou1 = test_iou[1]  # 更新最佳的测试集IoU第二个值
        best_epoch = epoch + 1  # 记录最佳精度所在的epoch
        # 保存当前最优模型
        torch.save(model.state_dict(), config['best_model_filename'])
        print(f"最佳模型已保存，测试集IoU第2个值：{best_test_iou1:.4f}，最佳epoch：{best_epoch}")

    # 保存模型
    if (epoch + 1) % config['save_interval'] == 0:
        save_path = f'train_result(NewModel_19_new_086)/model_{epoch + 1}.pth'
        torch.save(model.state_dict(), save_path)
        print(f'模型已保存至{save_path}')


    # 记录每轮的损失和评估指标到日志
    train_log_file.write(f"{epoch + 1}\t"
                         f"{avg_train_loss:.4f}\t"
                         f"{np.mean(train_iou[0]) if isinstance(train_iou, np.ndarray) else train_iou:.4f}\t"  # 第二个IoU值（排土场）
                         f"{np.mean(train_iou[1]) if isinstance(train_iou, np.ndarray) else train_iou:.4f}\t"  # 第三个IoU值（矿区）   
                         f"{train_precision[0] if isinstance(train_precision, np.ndarray) else train_precision:.4f}\t"  # 类别1 Precision（排土场）
                         f"{train_precision[1] if isinstance(train_precision, np.ndarray) else train_precision:.4f}\t"  # 类别2 Precision（矿区）
                         f"{train_recall[0] if isinstance(train_recall, np.ndarray) else train_recall:.4f}\t"  # 类别1 Recall（排土场）
                         f"{train_recall[1] if isinstance(train_recall, np.ndarray) else train_recall:.4f}\t"  # 类别2 Recall（矿区）
                         f"{train_f1_score[0] if isinstance(train_f1_score, np.ndarray) else train_f1_score:.4f}\t"  # 类别1 F1 Score（排土场）
                         f"{train_f1_score[1] if isinstance(train_f1_score, np.ndarray) else train_f1_score:.4f}\t"  # 类别2 F1 Score（矿区）
                         f"{np.mean(train_dice[0]):.4f}\t"
                         f"{np.mean(train_dice[1]):.4f}\t"
                         f"{train_Kappa:.4f}\t"
                         f"{train_overall_accuracy:.4f}\t"  # 总体准确率
                         f"{train_fw_iou:.4f}\n")

    test_log_file.write(f"{epoch + 1}\t"
                        f"{avg_test_loss:.4f}\t"
                        f"{np.mean(test_iou[0]) if isinstance(test_iou, np.ndarray) else test_iou:.4f}\t"  # 第二个IoU值（排土场）
                        f"{np.mean(test_iou[1]) if isinstance(test_iou, np.ndarray) else test_iou:.4f}\t"  # 第三个IoU值（矿区）
                        f"{test_precision[0] if isinstance(test_precision, np.ndarray) else test_precision:.4f}\t"  # 类别1 Precision（排土场）
                        f"{test_precision[1] if isinstance(test_precision, np.ndarray) else test_precision:.4f}\t"  # 类别2 Precision（矿区）
                        f"{test_recall[0] if isinstance(test_recall, np.ndarray) else test_recall:.4f}\t"  # 类别1 Recall（排土场）
                        f"{test_recall[1] if isinstance(test_recall, np.ndarray) else test_recall:.4f}\t"  # 类别2 Recall（矿区）
                        f"{test_f1_score[0] if isinstance(test_f1_score, np.ndarray) else test_f1_score:.4f}\t"  # 类别1 F1 Score（排土场）
                        f"{test_f1_score[1] if isinstance(test_f1_score, np.ndarray) else test_f1_score:.4f}\t"  # 类别2 F1 Score（矿区）
                        f"{np.mean(test_dice[0]):.4f}\t"
                        f"{np.mean(test_dice[1]):.4f}\t"
                        f"{test_Kappa:.4f}\t"
                        f"{test_overall_accuracy:.4f}\t"  # 总体准确率
                        f"{test_fw_iou:.4f}\n")

    # 重置评估器
    evaluator_train.reset()
    evaluator_test.reset()
# 计算并打印整个训练过程的总用时
total_end_time = time.time()
total_duration = total_end_time - total_start_time
print(f"整个训练过程的总用时：{total_duration:.2f}秒")
print(f"最佳模型位于第{best_epoch}轮，测试集精度为 {best_test_iou1:.4f}")  # 打印最佳模型的epoch
# 关闭日志文件
train_log_file.close()
test_log_file.close()