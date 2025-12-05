import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader
import json
import csv
import numpy as np

class TAGDataset(Dataset):
    def __init__(self, args, mode='train'):
        TAG_dir = 'tactile_datasets/TAG/dataset/'

        self.datalist = []
        self.labels = []
        self.sensor_type = []

        if mode == 'train':
            if args.dataset == 'rough':
                self.txt = 'tactile_datasets/TAG/train_rough.txt'
            elif args.dataset == 'material' or args.dataset == 'hard':
                self.txt = 'tactile_datasets/TAG/train.txt'
        else:
            if args.dataset == 'rough':
                self.txt = 'tactile_datasets/TAG/test_rough.txt'
            elif args.dataset == 'material' or args.dataset == 'hard':
                self.txt = 'tactile_datasets/TAG/test.txt'
        
        for line in open(self.txt):
            item = line.split(',')[0]
            label = int(line.split(',')[1])
            if label == -1:
                continue
            
            if args.dataset == 'hard':
                if label == 7 or label == 8 or label == 9 or label == 11 or label == 13:
                    label = 1
                else:
                    label = 0

            folder = item.split('/')[0]
            image = item.split('/')[1]
            self.datalist.append(TAG_dir + folder +'/gelsight_frame/'+ image)
            self.labels.append(label)
            self.sensor_type.append(0)
        
        if mode == 'train':
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        img = Image.open(self.datalist[index]).convert('RGB')
        
        img = self.transform(img)
        
        return img, self.sensor_type[index], self.labels[index]


class OBJ2Dataset(Dataset):
    def __init__(self, args, mode='train'):
        OBJ2_dir = 'tactile_datasets/obj2.0/touch/'

        self.datalist = []
        self.labels = []
        self.sensor_type = []

        self.mode = mode
        self.label_json_dir = 'tactile_datasets/obj2.0/label.json'
        self.split_json_dir = 'tactile_datasets/obj2.0/split.json'
        self.label_dict = {}
        
        with open(self.label_json_dir, 'r') as file:
            self.label_dict = json.load(file)

        with open(self.split_json_dir, 'r') as file:
            split_dict = json.load(file)
            samples_list = split_dict[mode]
            for item in samples_list:
                item_id = item[0]
                if int(item_id) <= 100:
                    continue
                png_id = item[1]
                self.datalist.append(OBJ2_dir + item_id +'/' + str(png_id) +'.png')
                self.labels.append(int(self.label_dict[item_id]))
                #print(item_id , int(self.label_dict[item_id]))
                #print(int(item_id), item_id)
                self.sensor_type.append(5)
        
        print(len(self.datalist))
        # print(self.labels)

        if mode == 'train':
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.RandomVerticalFlip(p=0.5),
                    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        img = Image.open(self.datalist[index]).convert('RGB')
        
        img = self.transform(img)
        
        return img, self.sensor_type[index], self.labels[index]

class OBJ1Dataset(Dataset):
    def __init__(self, args, mode='train'):
        OBJ1_dir = '/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/AnyTouch/tactile_datasets/ObjectFolder/ObjectFolder1.0/Objects/'

        self.datalist = []
        self.labels = []
        self.sensor_type = []

        self.mode = mode
        # shared json file with OF2.0 dataset
        self.label_json_dir = '/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/AnyTouch/tactile_datasets/label.json'
        self.split_json_dir = '/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/AnyTouch/tactile_datasets/split.json'
        self.label_dict = {}
        
        with open(self.label_json_dir, 'r') as file:
            self.label_dict = json.load(file)

        with open(self.split_json_dir, 'r') as file:
            split_dict = json.load(file)
            samples_list = split_dict[mode]
            for item in samples_list:
                item_id = item[0]
                if int(item_id) > 100:
                    continue
                png_id = item[1]
                self.datalist.append(OBJ1_dir + item_id +'/' + str(png_id) +'.png')
                self.labels.append(int(self.label_dict[item_id]))
                self.sensor_type.append(-1)
        
        print(len(self.datalist))
        # print(self.labels)
        if mode == 'train':
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        img = Image.open(self.datalist[index]).convert('RGB')
        
        img = self.transform(img)
        
        return img, self.sensor_type[index], self.labels[index]


class FeelDataset(Dataset):
    def __init__(self, args, mode='train'):
        TAG_dir = 'tactile_datasets/feel/'

        self.datalist = []
        self.labels = []
        self.sensor_type = []
        txt = open('tactile_datasets/feel/feel.csv', 'r')

        split_dict = np.load('tactile_datasets/feel/split_'+str(args.split)+'.npy', allow_pickle=True).item()
        name_list = split_dict[mode]

        csv_reader = csv.reader(txt)
        for row in csv_reader:
            name = row[0]
            png_id = row[1]

            if name in name_list:
                #self.datalist.append([TAG_dir + name +'/touch_after/'+str(png_id)+'_A.png', TAG_dir + name +'/touch_after/'+str(png_id)+'_B.png',TAG_dir + name +'/touch_before/'+str(png_id)+'_A.png', TAG_dir + name +'/touch_before/'+str(png_id)+'_B.png'])
                self.datalist.append([TAG_dir + name +'/touch_during/'+str(png_id)+'_A.png', TAG_dir + name +'/touch_during/'+str(png_id)+'_B.png'])
                self.labels.append(int(row[2]))
                self.sensor_type.append(0)

        print(len(self.datalist))
        
        if mode == 'train':
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224), antialias=True),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.3),
                    #transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224), antialias=True),
                    #transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        self.to_tensor = transforms.ToTensor()


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        img0 = Image.open(self.datalist[index][0]).convert('RGB')
        img1 = Image.open(self.datalist[index][1]).convert('RGB')

        img0 = self.to_tensor(img0).unsqueeze(0)
        img1 = self.to_tensor(img1).unsqueeze(0)

        img = torch.cat([img0, img1])
        touch = self.transform(img)
        
        return touch, self.sensor_type[index], self.labels[index]
