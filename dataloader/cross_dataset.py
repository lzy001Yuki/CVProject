import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader
import csv
import json

from matplotlib import pyplot as plt
import numpy as np
import random

SSVTP_dir = 'tactile_datasets/SSVTP/ssvtp_data/images_tac/'
TAG_dir = 'tactile_datasets/TAG/dataset/'
obreal_dir = 'tactile_datasets/objectfolder/real/tactile/'
visgel_dir = 'tactile_datasets/visgel/images/touch/'
yuan18_dir = 'tactile_datasets/yuan18/Data_ICRA18/Data/'
TVL_dir = 'tactile_datasets/TVL/tvl_dataset/hct/'
ycb_dir = 'tactile_datasets/YCB-Slide/real/'
octopi_dir = 'tactile_datasets/octopi/processed/'

TAG_file = 'tactile_datasets/TAG/label.txt'
obreal_file = 'tactile_datasets/objectfolder/real/contact_obj.csv'
visgel_file = 'tactile_datasets/visgel/images/contact_visgel.csv'
yuan18_file = 'tactile_datasets/yuan18/Data_ICRA18/contact_yuan.csv'
octopi_file = 'tactile_datasets/octopi/contact.csv'

feel_dir = 'tactile_datasets/feel/'
feel_file = 'tactile_datasets/feel/feel.csv'
obj2_dir = 'tactile_datasets/obj2.0/'

tacquad_indoor_dir = 'tactile_datasets/tacquad/data_indoor/'
tacquad_outdoor_dir = 'tactile_datasets/tacquad/data_outdoor/'

tacquad_indoor_file = 'tactile_datasets/tacquad/contact_indoor.csv'
tacquad_outdoor_file = 'tactile_datasets/tacquad/contact_outdoor.csv'

def custom_sort(a):
    int_a = int(a.split('.')[0])
    return int_a

def custom_sort_visgel(a):
    a0 = a.split('.')[0]
    int_a = int(a0.split('e')[1])
    return int_a

class PretrainDataset_cross(Dataset):
    def __init__(self, mode='train'):

        self.datalist = []
        self.sensor_type = []
        self.objectlist = []
        self.object_index_pairs = []

        obj_id = -1

        with open(tacquad_indoor_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                item_name = row[0]
                obj_id += 1

                if obj_id > 0:
                    self.object_index_pairs[-1].append(len(self.datalist)-1)

                self.object_index_pairs.append([len(self.datalist)])


                gelsight_start = int(row[1])
                gelsight_end = int(row[2])

                digit_start = int(row[3])
                digit_end = int(row[4])

                duragel_start = int(row[5])
                duragel_end = int(row[6])

                for t in range(gelsight_start + 3, gelsight_end+1):
                    png_path = tacquad_indoor_dir + item_name +'/gelsight/' + str(t) +'.png'
                    self.datalist.append(png_path)
                    self.objectlist.append(obj_id)
                    self.sensor_type.append(3)
                
                for t in range(digit_start + 3, digit_end+1):
                    png_path = tacquad_indoor_dir + item_name +'/digit/' + str(t) +'.png'
                    self.datalist.append(png_path)
                    self.objectlist.append(obj_id)
                    self.sensor_type.append(1)

                for t in range(duragel_start + 3, duragel_end+1):
                    png_path = tacquad_indoor_dir + item_name +'/duragel/' + str(t) +'.png'
                    self.datalist.append(png_path)
                    self.objectlist.append(obj_id)
                    self.sensor_type.append(4)

        with open(tacquad_outdoor_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                item_name = row[0]
                obj_id += 1

                if obj_id > 0:
                    self.object_index_pairs[-1].append(len(self.datalist)-1)

                self.object_index_pairs.append([len(self.datalist)])

                gelsight_start = int(row[1])
                gelsight_end = int(row[2])

                digit_start = int(row[3])
                digit_end = int(row[4])

                duragel_start = int(row[5])
                duragel_end = int(row[6])

                for t in range(gelsight_start + 3, gelsight_end+1):
                    png_path = tacquad_outdoor_dir + item_name +'/gelsight/' + str(t) +'.png'
                    self.datalist.append(png_path)
                    self.objectlist.append(obj_id)
                    self.sensor_type.append(3)
                
                for t in range(digit_start + 3, digit_end+1):
                    png_path = tacquad_outdoor_dir + item_name +'/digit/' + str(t) +'.png'
                    self.datalist.append(png_path)
                    self.objectlist.append(obj_id)
                    self.sensor_type.append(1)

                for t in range(duragel_start + 3, duragel_end+1):
                    png_path = tacquad_outdoor_dir + item_name +'/duragel/' + str(t) +'.png'
                    self.datalist.append(png_path)
                    self.objectlist.append(obj_id)
                    self.sensor_type.append(4)
        self.object_index_pairs[-1].append(len(self.datalist)-1)

        with open(obreal_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.datalist.append(row[0])
                self.sensor_type.append(2)


        with open(visgel_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.datalist.append(row[0])
                self.sensor_type.append(0)

        with open(yuan18_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.datalist.append(row[0])
                self.sensor_type.append(0)
        
        with open(octopi_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.datalist.append(octopi_dir+row[0])
                self.sensor_type.append(3)

        with open(TAG_file,'r') as file:
            for row in file:
                image = row.split(',')[0]
                folder = image.split('/')[0]
                image_id = image.split('/')[1]
                for tt in range(1):
                    self.datalist.append(TAG_dir + folder + '/gelsight_frame/' + image_id)
                    self.sensor_type.append(0)


        for item in os.listdir(SSVTP_dir):
            self.datalist.append(SSVTP_dir+item)
            self.sensor_type.append(1)

        for data_folder in ['data1/','data2/','data3/']:
            now_data_folder = TVL_dir + data_folder
            now_json = now_data_folder + 'contact.json'
            
            with open(now_json, 'r') as file:
                contact_json = json.load(file)
                for image in contact_json['tactile']:
                    self.datalist.append(now_data_folder + image)
                    self.sensor_type.append(1)

        for folder in os.listdir(ycb_dir):
            now_folder = ycb_dir + folder + '/'
            if os.path.isdir(now_folder):
                for now_data in ['dataset_0','dataset_1','dataset_2','dataset_3','dataset_4']:
                    now_image_folder = now_folder + now_data + '/frames/'
                    for image in os.listdir(now_image_folder):
                        self.datalist.append(now_image_folder + image)
                        self.sensor_type.append(1)
        
        print(len(self.datalist), len(self.sensor_type), len(self.objectlist), len(self.object_index_pairs))

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
        return len(self.objectlist)

    def __getitem__(self, index):

        img = Image.open(self.datalist[index]).convert('RGB')
        
        img = self.transform(img)

        for index_pair in self.object_index_pairs:
            index_start = index_pair[0]
            index_end = index_pair[1]

            if index_start<=index<=index_end:
                break

        positive_index = random.randint(index_start, index_end)
        while self.sensor_type[positive_index] == self.sensor_type[index]:
            positive_index = random.randint(index_start, index_end)

        positive = Image.open(self.datalist[positive_index]).convert('RGB')
        positive = self.transform(positive)


        negative_index = random.randint(0, len(self.datalist)-1)
        while negative_index < len(self.objectlist):
            if self.objectlist[negative_index] != self.objectlist[index]:
                break
            negative_index = random.randint(0, len(self.datalist)-1)

        negative = Image.open(self.datalist[negative_index]).convert('RGB')
        negative = self.transform(negative)

        # print(self.datalist[index],self.datalist[positive_index],self.datalist[negative_index])
        
        return img, self.sensor_type[index], positive, self.sensor_type[positive_index], negative, self.sensor_type[negative_index]

class PretrainDataset_cross_video(Dataset):
    def __init__(self, mode='train'):

        self.datalist = []
        self.sensor_type = []
        self.objectlist = []
        self.object_index_pairs = []

        obj_id = -1

        with open(tacquad_indoor_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                item_name = row[0]
                obj_id+=1

                if obj_id > 0:
                    self.object_index_pairs[-1].append(len(self.datalist)-1)

                self.object_index_pairs.append([len(self.datalist)])

                gelsight_start = int(row[1])
                gelsight_end = int(row[2])

                digit_start = int(row[3])
                digit_end = int(row[4])

                duragel_start = int(row[5])
                duragel_end = int(row[6])

                for t in range(gelsight_start + 3, gelsight_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_indoor_dir + item_name +'/gelsight/' + str(t) +'.png'
                    png_path_2 = tacquad_indoor_dir + item_name +'/gelsight/' + str(t-1) +'.png'
                    png_path_1 = tacquad_indoor_dir + item_name +'/gelsight/' + str(t-2) +'.png'
                    self.datalist.append([png_path_1, png_path_2, png_path_3])
                    self.objectlist.append(obj_id)
                    self.sensor_type.append(3)
                
                for t in range(digit_start + 3, digit_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_indoor_dir + item_name +'/digit/' + str(t) +'.png'
                    png_path_2 = tacquad_indoor_dir + item_name +'/digit/' + str(t-1) +'.png'
                    png_path_1 = tacquad_indoor_dir + item_name +'/digit/' + str(t-2) +'.png'
                    self.datalist.append([png_path_1, png_path_2, png_path_3])
                    self.objectlist.append(obj_id)
                    self.sensor_type.append(1)

                for t in range(duragel_start + 3, duragel_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_indoor_dir + item_name +'/duragel/' + str(t) +'.png'
                    png_path_2 = tacquad_indoor_dir + item_name +'/duragel/' + str(t-1) +'.png'
                    png_path_1 = tacquad_indoor_dir + item_name +'/duragel/' + str(t-2) +'.png'
                    self.datalist.append([png_path_1, png_path_2, png_path_3])
                    self.objectlist.append(obj_id)
                    self.sensor_type.append(4)

        with open(tacquad_outdoor_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                item_name = row[0]
                obj_id+=1

                if obj_id > 0:
                    self.object_index_pairs[-1].append(len(self.datalist)-1)

                self.object_index_pairs.append([len(self.datalist)])

                gelsight_start = int(row[1])
                gelsight_end = int(row[2])

                digit_start = int(row[3])
                digit_end = int(row[4])

                duragel_start = int(row[5])
                duragel_end = int(row[6])

                for t in range(gelsight_start + 3, gelsight_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_outdoor_dir + item_name +'/gelsight/' + str(t) +'.png'
                    png_path_2 = tacquad_outdoor_dir + item_name +'/gelsight/' + str(t-1) +'.png'
                    png_path_1 = tacquad_outdoor_dir + item_name +'/gelsight/' + str(t-2) +'.png'
                    self.datalist.append([png_path_1, png_path_2, png_path_3])
                    self.objectlist.append(obj_id)
                    self.sensor_type.append(3)
                
                for t in range(digit_start + 3, digit_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_outdoor_dir + item_name +'/digit/' + str(t) +'.png'
                    png_path_2 = tacquad_outdoor_dir + item_name +'/digit/' + str(t-1) +'.png'
                    png_path_1 = tacquad_outdoor_dir + item_name +'/digit/' + str(t-2) +'.png'
                    self.datalist.append([png_path_1, png_path_2, png_path_3])
                    self.objectlist.append(obj_id)
                    self.sensor_type.append(1)

                for t in range(duragel_start + 3, duragel_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_outdoor_dir + item_name +'/duragel/' + str(t) +'.png'
                    png_path_2 = tacquad_outdoor_dir + item_name +'/duragel/' + str(t-1) +'.png'
                    png_path_1 = tacquad_outdoor_dir + item_name +'/duragel/' + str(t-2) +'.png'
                    self.datalist.append([png_path_1, png_path_2, png_path_3])
                    self.objectlist.append(obj_id)
                    self.sensor_type.append(4)
        self.object_index_pairs[-1].append(len(self.datalist)-1)

        with open(obreal_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                image_id = int(row[0].split('gelsight/')[1].split('.')[0])
                if image_id >= 3:
                    image_1 = str(image_id - 2) +'.png'
                    image_2 = str(image_id - 1) +'.png'
                    now_folder = row[0].split('gelsight/')[0] + 'gelsight/'
                    self.datalist.append([now_folder + image_1, now_folder + image_2, row[0]])
                    self.sensor_type.append(2)


        with open(visgel_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                image_id = int(row[0].split('/frame')[1].split('.')[0])
                if image_id >= 3:
                    image_1 = 'frame' + str(image_id - 2).zfill(4) +'.jpg'
                    image_2 = 'frame' + str(image_id - 1).zfill(4) +'.jpg'
                    now_folder = row[0].split('/frame')[0] + '/'
                    self.datalist.append([now_folder + image_1, now_folder + image_2, row[0]])
                    self.sensor_type.append(0)

        with open(yuan18_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                image_id = int(row[0].split('gelsight_frame/')[1].split('.')[0])
                if image_id >= 3:
                    image_1 = str(image_id - 2).zfill(4) +'.png'
                    image_2 = str(image_id - 1).zfill(4) +'.png'
                    now_folder = row[0].split('gelsight_frame/')[0] + 'gelsight_frame/'
                    self.datalist.append([now_folder + image_1, now_folder + image_2, row[0]])
                    self.sensor_type.append(0)

        with open(TAG_file,'r') as file:
            for row in file:
                image = row.split(',')[0]
                folder = image.split('/')[0]
                image_name = image.split('/')[1]
                now_folder = TAG_dir + folder + '/gelsight_frame/'
                image_id = int(image_name.split('.')[0])
                if image_id >= 3:
                    image_1 = str(image_id - 2).zfill(10) +'.jpg'
                    image_2 = str(image_id - 1).zfill(10) +'.jpg'
                    for tt in range(1):
                        self.datalist.append([now_folder + image_1, now_folder + image_2, now_folder + image_name])
                        self.sensor_type.append(0)
        
        # TVL
        for data_folder in ['data1/','data2/','data3/']:
            now_data_folder = TVL_dir + data_folder
            now_json = now_data_folder + 'contact.json'
            
            with open(now_json, 'r') as file:
                contact_json = json.load(file)
                for image in contact_json['tactile']:
                    image_id = int(image.split('/')[2].split('-')[0])
                    image_list = os.listdir(now_data_folder + image.split('/')[0]+'/tactile')
                    image_0 = None
                    image_1 = None
                    image_2 = None
                    for file in image_list:
                        if file.startswith(str(image_id-2)):
                            image_1 = file
                        elif file.startswith(str(image_id-1)):
                            image_2 = file
                        
                        if image_1 and image_2:
                            break

                    if image_1 and image_2:
                        now_image_folder = now_data_folder + image.split('/')[0]+'/tactile/'
                        if os.path.exists(now_image_folder + image_1) and os.path.exists(now_data_folder + image):
                            self.datalist.append([now_image_folder + image_1, now_image_folder + image_2, now_data_folder + image])
                            self.sensor_type.append(1)


        for folder in os.listdir(ycb_dir):
            now_folder = ycb_dir + folder + '/'
            if os.path.isdir(now_folder):
                for now_data in ['dataset_0','dataset_1','dataset_2','dataset_3','dataset_4']:
                    now_image_folder = now_folder + now_data + '/frames/'
                    for image in os.listdir(now_image_folder):
                        image_id = int(image.split('_')[1].split('.')[0])
                        if image_id >= 9:
                            image_1 = 'frame_' + str(image_id - 6).zfill(7) +'.jpg'
                            image_2 = 'frame_' + str(image_id - 3).zfill(7) +'.jpg'
                            if os.path.exists(now_image_folder + image_1) and os.path.exists(now_image_folder + image):
                                self.datalist.append([now_image_folder + image_1, now_image_folder + image_2, now_image_folder + image])
                                self.sensor_type.append(1)
        
        with open(octopi_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                image = row[0]
                folder = image.split('/')[1]
                image_name = image.split('/')[2]

                now_folder = octopi_dir + folder + '/'

                image_id = int(image_name.split('.')[0])
                if image_id >= 3:
                    image_1 = str(image_id - 2).zfill(10) +'.jpg'
                    image_2 = str(image_id - 1).zfill(10) +'.jpg'
                    self.datalist.append([now_folder + image_1, now_folder + image_2, now_folder + image_name])
                    self.sensor_type.append(3)

        print(len(self.datalist), len(self.sensor_type), len(self.objectlist), len(self.object_index_pairs))

        if mode == 'train':
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224), antialias=False),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.3),
                    # transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224), antialias=False),
                    # transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.objectlist)

    def __getitem__(self, index):

        img1 = Image.open(self.datalist[index][0]).convert('RGB')
        img2 = Image.open(self.datalist[index][1]).convert('RGB')
        img3 = Image.open(self.datalist[index][2]).convert('RGB')

        img1 = self.to_tensor(img1).unsqueeze(0)
        img2 = self.to_tensor(img2).unsqueeze(0)
        img3 = self.to_tensor(img3).unsqueeze(0)
        img = torch.cat([img1, img2, img3])
        img = self.transform(img)

        for index_pair in self.object_index_pairs:
            index_start = index_pair[0]
            index_end = index_pair[1]

            if index_start<=index<=index_end:
                break

        positive_index = random.randint(index_start, index_end)
        while self.sensor_type[positive_index] == self.sensor_type[index]:
            positive_index = random.randint(index_start, index_end)

        img1 = Image.open(self.datalist[positive_index][0]).convert('RGB')
        img2 = Image.open(self.datalist[positive_index][1]).convert('RGB')
        img3 = Image.open(self.datalist[positive_index][2]).convert('RGB')

        img1 = self.to_tensor(img1).unsqueeze(0)
        img2 = self.to_tensor(img2).unsqueeze(0)
        img3 = self.to_tensor(img3).unsqueeze(0)
        positive = torch.cat([img1, img2, img3])
        positive = self.transform(positive)

        negative_index = random.randint(0, len(self.datalist)-1)
        while negative_index < len(self.objectlist):
            if self.objectlist[negative_index] != self.objectlist[index]:
                break
            negative_index = random.randint(0, len(self.datalist)-1)

        img1 = Image.open(self.datalist[negative_index][0]).convert('RGB')
        img2 = Image.open(self.datalist[negative_index][1]).convert('RGB')
        img3 = Image.open(self.datalist[negative_index][2]).convert('RGB')

        img1 = self.to_tensor(img1).unsqueeze(0)
        img2 = self.to_tensor(img2).unsqueeze(0)
        img3 = self.to_tensor(img3).unsqueeze(0)
        negative = torch.cat([img1, img2, img3])
        negative = self.transform(negative)
        
        # print(self.datalist[index],self.datalist[positive_index],self.datalist[negative_index])

        return img, self.sensor_type[index], positive, self.sensor_type[positive_index], negative, self.sensor_type[negative_index]

