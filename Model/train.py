
import torch
import time  
import os.path
import os
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from Dataset_2 import MyDataset  
from NewModel_19 import  FusionNetVGG
from tqdm import tqdm
from metric import Evaluator 
from set_seed import set_seed 
from CosineAnnealingLR import CosineAnnealingWarmupRestarts
from loss_fn21_1 import  DynamicMiningLoss

set_seed(1) 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

config = {
    'train_image_dir': 'data/1-train/1-images',  
    'train_label_dir': 'data/1-train/2-labels',
    'test_image_dir': 'data/2-test/1-images',
    'test_label_dir': 'data/2-test/2-labels',
    'batch_size': 10, 
    'n_channels': 5,  
    'num_classes': 2,  
    'learning_rate': 0.001,  
    'num_epochs': 600, 
    'save_interval': 150,  
    'log_dir': 'train_result(NewModel_19_new_086)', 
    'train_log_filename': 'train_result(NewModel_19_new_086)/train_log.txt', 
    'test_log_filename': 'train_result(NewModel_19_new_086)/test_log.txt', 
    'best_model_filename': 'train_result(NewModel_19_new_086)/best_model.pth',  
    'first_cycle_steps': 100, 
    'cycle_mult': 1.5, 
    'max_lr': 0.001, 
    'min_lr': 1e-5,  
    'warmup_steps': 20,
    'gamma': 0.98,
    'use_augmentation': True}

train_dataset = MyDataset(config['train_image_dir'], config['train_label_dir'], transform=config['use_augmentation'])
test_dataset = MyDataset(config['test_image_dir'], config['test_label_dir'],transform=False )

train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print(f'训练数据集的长度为：{train_dataset_size}')
print(f'测试数据集的长度为：{test_dataset_size}')

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=8,         
    pin_memory=True,         
    persistent_workers=True  
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

model = FusionNetVGG(num_classes=2).to(device)
loss_fn = DynamicMiningLoss(
    alpha=0.86,    
    power=0.5    
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=config['first_cycle_steps'],
                                          cycle_mult=config['cycle_mult'],
                                          max_lr=config['max_lr'],
                                          min_lr=config['min_lr'],
                                          warmup_steps=config['warmup_steps'],
                                          gamma=config['gamma'])

if not os.path.exists(config['log_dir']):
    os.makedirs(config['log_dir'])

evaluator_train = Evaluator(num_class=config['num_classes']) 
evaluator_test = Evaluator(num_class=config['num_classes']) 

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
                     "Train_F1_Score_Class_1\t"  
                     "Train_F1_Score_Class_2\t"  
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
                    "test_F1_Score_Class_1\t" 
                    "test_F1_Score_Class_2\t"  
                    "test_Dice_Class_1\t"
                    "test_Dice_Class_2\t"
                    "test_Kappa\t"
                    "test_OA\t"
                    "test_FWIoU\n")

best_test_iou1 = -1.0 

total_start_time = time.time()
total_train_step = 0  
total_test_step = 0  

best_epoch = 0  

for epoch in range(config['num_epochs']):
    model.train()  
    epoch_start_time = time.time()  
    total_train_loss = 0  
    total_batches = 0  

    pbar = tqdm(train_dataloader, desc=f"训练第{epoch + 1}轮", ncols=100)  
    for data in pbar:
        (input_branch_1, input_branch_2),labels = data
        input_branch_1, input_branch_2, labels =input_branch_1.to(device), input_branch_2.to(device), labels.to(device)

        outputs = model(input_branch_1, input_branch_2)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
        total_train_step += 1 

        total_train_loss += loss.item() 
        total_batches += 1 
        current_lr = optimizer.param_groups[0]['lr']

        pbar.set_postfix(loss=total_train_loss / total_batches, lr=current_lr)
        pred_train = outputs.argmax(dim=1).cpu().numpy()
        gt_train = labels.cpu().numpy()
        evaluator_train.add_batch(gt_train, pred_train)  
    avg_train_loss = total_train_loss / total_batches
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"第{epoch + 1}轮训练完成，训练时间：{epoch_duration:.2f}秒")
    print(f"第{epoch + 1}轮训练的平均损失：{avg_train_loss:.4f}")
    print(f"第{epoch + 1}轮训练的学习率为：{current_lr:.5f}")
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

    scheduler.step(epoch)
    model.eval() 
    total_test_loss = 0  

    with torch.no_grad(): 
        for data in test_dataloader:
            (input_branch_1, input_branch_2), labels = data
            input_branch_1, input_branch_2, labels = input_branch_1.to(device), input_branch_2.to(device), labels.to(
                device)
            outputs = model(input_branch_1, input_branch_2) 
            loss = loss_fn(outputs, labels)
            total_test_loss += loss.item() 
            pred_test = outputs.argmax(dim=1).cpu().numpy()
            gt_test = labels.cpu().numpy()
            evaluator_test.add_batch(gt_test, pred_test)
    avg_test_loss = total_test_loss / len(test_dataloader)
    print(f"第{epoch + 1}轮测试完成，平均损失为：{avg_test_loss:.2f}")
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
    if test_iou[1] > best_test_iou1:
        best_test_iou1 = test_iou[1]  
        best_epoch = epoch + 1 
        torch.save(model.state_dict(), config['best_model_filename'])
        print(f"最佳模型已保存，测试集IoU第2个值：{best_test_iou1:.4f}，最佳epoch：{best_epoch}")
    if (epoch + 1) % config['save_interval'] == 0:
        save_path = f'train_result(NewModel_19_new_086)/model_{epoch + 1}.pth'
        torch.save(model.state_dict(), save_path)
        print(f'模型已保存至{save_path}')

    train_log_file.write(f"{epoch + 1}\t"
                         f"{avg_train_loss:.4f}\t"
                         f"{np.mean(train_iou[0]) if isinstance(train_iou, np.ndarray) else train_iou:.4f}\t" 
                         f"{np.mean(train_iou[1]) if isinstance(train_iou, np.ndarray) else train_iou:.4f}\t"  
                         f"{train_precision[0] if isinstance(train_precision, np.ndarray) else train_precision:.4f}\t" 
                         f"{train_precision[1] if isinstance(train_precision, np.ndarray) else train_precision:.4f}\t" 
                         f"{train_recall[0] if isinstance(train_recall, np.ndarray) else train_recall:.4f}\t"  
                         f"{train_recall[1] if isinstance(train_recall, np.ndarray) else train_recall:.4f}\t" 
                         f"{train_f1_score[0] if isinstance(train_f1_score, np.ndarray) else train_f1_score:.4f}\t" 
                         f"{train_f1_score[1] if isinstance(train_f1_score, np.ndarray) else train_f1_score:.4f}\t" 
                         f"{np.mean(train_dice[0]):.4f}\t"
                         f"{np.mean(train_dice[1]):.4f}\t"
                         f"{train_Kappa:.4f}\t"
                         f"{train_overall_accuracy:.4f}\t" 
                         f"{train_fw_iou:.4f}\n")
    test_log_file.write(f"{epoch + 1}\t"
                        f"{avg_test_loss:.4f}\t"
                        f"{np.mean(test_iou[0]) if isinstance(test_iou, np.ndarray) else test_iou:.4f}\t"  
                        f"{np.mean(test_iou[1]) if isinstance(test_iou, np.ndarray) else test_iou:.4f}\t"  
                        f"{test_precision[0] if isinstance(test_precision, np.ndarray) else test_precision:.4f}\t" 
                        f"{test_precision[1] if isinstance(test_precision, np.ndarray) else test_precision:.4f}\t"  
                        f"{test_recall[0] if isinstance(test_recall, np.ndarray) else test_recall:.4f}\t"  
                        f"{test_recall[1] if isinstance(test_recall, np.ndarray) else test_recall:.4f}\t" 
                        f"{test_f1_score[0] if isinstance(test_f1_score, np.ndarray) else test_f1_score:.4f}\t"  
                        f"{test_f1_score[1] if isinstance(test_f1_score, np.ndarray) else test_f1_score:.4f}\t" 
                        f"{np.mean(test_dice[0]):.4f}\t"
                        f"{np.mean(test_dice[1]):.4f}\t"
                        f"{test_Kappa:.4f}\t"
                        f"{test_overall_accuracy:.4f}\t"  
                        f"{test_fw_iou:.4f}\n")

    evaluator_train.reset()
    evaluator_test.reset()
total_end_time = time.time()
total_duration = total_end_time - total_start_time
print(f"整个训练过程的总用时：{total_duration:.2f}秒")
print(f"最佳模型位于第{best_epoch}轮，测试集精度为 {best_test_iou1:.4f}") 
train_log_file.close()
test_log_file.close()
