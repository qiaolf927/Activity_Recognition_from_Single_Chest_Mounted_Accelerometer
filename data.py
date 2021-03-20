import numpy as np
import torch
import torch.utils.data

class TrainValDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len, csv_files):
        self.seq_len = seq_len
        self.csv_files = csv_files
        self.freq = 10  # 表示采样频率,10表示每10行取一个数据
        self.stride = 1 #表示几行数据合为1
        self.mean = 2000 #标准化下输入
        self.var = 25.0

        data_record = {} #构建字典存储某一个csv文件的数据
        data_num = {} #构建字典存储某一个csv文件的数据点个数
        seq_data_num = 0 #记录总共有多少个
        file_refer = [] #数据index与文件的索引
        file_offset = [0] #数据index对于文件的偏置,需要减去这个偏置再从相应的文件中算出index
        for index, file in enumerate(csv_files):
            data = []
            with open(file, 'r') as f:
                num = 0
                freq_num = 0
                for line in f.readlines():
                    if freq_num < self.freq - 1:
                        freq_num += 1
                        continue
                    freq_num = 0
                    item = line.split(',')
                    if item[0] == '':
                        break
                    if self.stride == 1:
                        item_data = [(float(item[1]) - self.mean) / self.var, (float(item[2]) - self.mean) / self.var,
                                     (float(item[3]) - self.mean) / self.var, int(item[4][0])]
                        data.append(item_data)
                    else:
                        if num == 0:
                            item_data = [(float(item[1]) - self.mean) / self.var, (float(item[2]) - self.mean) / self.var, (float(item[3]) - self.mean) / self.var]  # 按逗号分，最后一个字符串包含\n
                            num += 1
                        elif num == self.stride - 1:
                            item_data.append((float(item[1]) - self.mean) / self.var)
                            item_data.append((float(item[2]) - self.mean) / self.var)
                            item_data.append((float(item[3]) - self.mean) / self.var)
                            item_data.append(int(item[4][0]))
                            data.append(item_data)
                            num = 0
                        else:
                            item_data.append((float(item[1]) - self.mean) / self.var)
                            item_data.append((float(item[2]) - self.mean) / self.var)
                            item_data.append((float(item[3]) - self.mean) / self.var)
                            num += 1

            data = np.array(data)
            data_record[file] = data #保存此file的数据
            data_num[file] = data.shape[0]
            file_seq_data_num = int(np.ceil(data.shape[0] / seq_len))
            seq_data_num += file_seq_data_num
            for i in range(file_seq_data_num):
                file_refer.append(index)
            file_offset.append(file_offset[-1] + file_seq_data_num)

        self.data_record = data_record
        self.data_num = data_num
        self.seq_data_num = seq_data_num
        self.file_refer = file_refer
        self.file_offset = file_offset


    def __len__(self):
        return self.seq_data_num

    def __getitem__(self, index):

        fileindex = self.file_refer[index]
        filename = self.csv_files[fileindex]
        offset = self.file_offset[fileindex]
        index_in_file = index - offset
        filedata = self.data_record[filename]

        feats = []
        labels = []
        flags = [] #表示该数据点需要被预测
        for i in range(self.seq_len):
            curr_data_point = index_in_file * self.seq_len + i #具体数据点位置
            #超出边界点的部分，取用边界点的数据
            flag = 1
            if curr_data_point < 0:
                curr_data_point = 0
                flag = 0
            if curr_data_point > self.data_num[filename] - 1:
                curr_data_point = self.data_num[filename] - 1
                flag = 0
            feat = [filedata[curr_data_point][j] for j in range(self.stride*3)]
            feat = torch.Tensor(feat)
            label = filedata[curr_data_point][self.stride*3]
            feats.append(feat)
            labels.append(label)
            flags.append(flag)
        feats = torch.stack(feats)
        labels = np.array(labels)
        flags = np.array(flags)
        return feats, labels, flags