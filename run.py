import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

import model as Model
import data as data
import torch_utils as torch_utils



def train(model, train_loader, optimizer, scheduler, epoch, criterion):
    num_batches = len(train_loader)
    model.train()
    for batch_idx, (feats, target, _) in enumerate(train_loader):
        target = target.view(-1)
        target = target.long()
        optimizer.zero_grad()
        output = model(feats)
        output = output.view(-1, 8) #类别个数
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print('epoch {}, iter {}/{}: loss = {}'.format(epoch, batch_idx + 1, num_batches, loss.item()))
    scheduler.step()


def val(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    batch_num = 0
    total_count = 0
    correct = 0
    with torch.no_grad():
        for feats, target, flags in test_loader:
            batch_num += 1
            feats = feats
            target = target.view(-1)
            target = target.long()
            flags = flags.view(-1)
            output = model(feats)
            output = output.view(-1, 8)
            loss = criterion(output, target)

            test_loss += loss.item()
            curr_output = output
            curr_output = curr_output.view(-1, 8)
            curr_output = F.softmax(curr_output, dim=1)
            curr_output = torch_utils.to_numpy(curr_output)
            curr_infer = np.argmax(curr_output, axis=1)

            curr_target = target
            curr_target = torch_utils.to_numpy(curr_target)
            flags = torch_utils.to_numpy(flags)
            for infer_item, target_item, flag in zip(curr_infer, curr_target, flags):
                # print([infer_item, target_item, flag])
                if flag > 0.5:
                    total_count += 1
                    if infer_item == target_item:
                        correct += 1

    test_loss /= batch_num
    acc = correct/total_count
    print('acc: {:.4f}, loss: {:.4f}'.format(acc, test_loss))
    return acc


def main():
    # 数据集划分
    trainsplit = ['1.csv', '2.csv', '3.csv', '4.csv', '5.csv', '6.csv', '7.csv', '8.csv', '9.csv', '10.csv']
    valsplit = ['11.csv', '12.csv', '13.csv', '14.csv', '15.csv']
    seq_len = 32
    batch_size = 16

    # 加载数据
    train_dataset = data.TrainValDataset(seq_len, trainsplit)
    val_dataset = data.TrainValDataset(seq_len, valsplit)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=32, pin_memory=True, drop_last=False)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=True, num_workers=32, pin_memory=True, drop_last=False)
    model = Model.toy_LSTM(seq_len)

    #优化器
    # optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.SGD(model.parameters(), lr=1e-1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 2000])
    criterion = nn.CrossEntropyLoss()
    print("...data and model loaded")

    ckpt_step = 50
    epochs = 3000
    max_acc = 0
    max_pos = 0

    print("...begin training")

    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, scheduler, epoch, criterion)
        torch_utils.save_checkpoint({
            'state_dict': model.state_dict(), 'epoch': epoch + 1,}, is_best=False, fpath='ckpt/checkpoint_epoch_{}.pth.tar'.format(epoch))

        metric = val(model, val_loader, criterion)
        if metric > max_acc:
            max_acc = metric
            max_pos = epoch
            src_path = 'ckpt/checkpoint_epoch_{}.pth.tar'.format(epoch)
            dst_path = 'ckpt/best.pth.tar'
            if os.path.exists(dst_path):
                os.remove(dst_path)
            shutil.copy(src_path, dst_path)
        print('curr {:.6f}, max {:.6f} at epoch {}'.format(metric, max_acc, max_pos))

        if epoch % ckpt_step == 0 or epoch == epochs:
            print('...removing unnecessary ckpts')
            beg = epoch - ckpt_step + 1
            end = epoch + 1
            for i in range(beg, end):
                path = 'ckpt/checkpoint_epoch_{}.pth.tar'.format(i)
                os.remove(path)


    print("...end training")


if __name__ == '__main__':
    main()