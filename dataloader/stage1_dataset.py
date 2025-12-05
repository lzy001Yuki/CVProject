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

SSVTP_dir = 'tactile_datasets/TVL/tvl_dataset/ssvtp/images_tac/'
TAG_dir = 'tactile_datasets/TAG/dataset/'
obreal_dir = 'tactile_datasets/objectfolder/real/tactile/'
visgel_dir = 'tactile_datasets/visgel/images/touch/'
yuan18_dir = 'tactile_datasets/yuan18/Data_ICRA18/Data/'
TVL_dir = 'tactile_datasets/TVL/tvl_dataset/hct/'
ycb_dir = 'tactile_datasets/YCB-Slide/real/'
octopi_dir = 'tactile_datasets/octopi/'

TAG_file = 'tactile_datasets/TAG/label.txt'
obreal_file = 'tactile_datasets/contact_obj.csv'
visgel_file = 'tactile_datasets/contact_visgel.csv'
yuan18_file = 'tactile_datasets/contact_yuan.csv'
octopi_file = 'tactile_datasets/contact_octopi.csv'

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


class PretrainDataset_Contact(Dataset):
    def __init__(self, mode='train'):

        self.datalist = []
        self.sensor_type = []

        with open(tacquad_indoor_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                item_name = row[0]

                gelsight_start = int(row[1])
                gelsight_end = int(row[2])

                digit_start = int(row[3])
                digit_end = int(row[4])

                duragel_start = int(row[5])
                duragel_end = int(row[6])

                for t in range(gelsight_start, gelsight_end+1):
                    png_path = tacquad_indoor_dir + item_name +'/gelsight/' + str(t) +'.png'
                    self.datalist.append(png_path)
                    self.sensor_type.append(3)
                
                for t in range(digit_start, digit_end+1):
                    png_path = tacquad_indoor_dir + item_name +'/digit/' + str(t) +'.png'
                    self.datalist.append(png_path)
                    self.sensor_type.append(1)

                for t in range(duragel_start, duragel_end+1):
                    png_path = tacquad_indoor_dir + item_name +'/duragel/' + str(t) +'.png'
                    self.datalist.append(png_path)
                    self.sensor_type.append(4)

        with open(tacquad_outdoor_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                item_name = row[0]

                gelsight_start = int(row[1])
                gelsight_end = int(row[2])

                digit_start = int(row[3])
                digit_end = int(row[4])

                duragel_start = int(row[5])
                duragel_end = int(row[6])

                for t in range(gelsight_start, gelsight_end+1):
                    png_path = tacquad_outdoor_dir + item_name +'/gelsight/' + str(t) +'.png'
                    self.datalist.append(png_path)
                    self.sensor_type.append(3)
                
                for t in range(digit_start, digit_end+1):
                    png_path = tacquad_outdoor_dir + item_name +'/digit/' + str(t) +'.png'
                    self.datalist.append(png_path)
                    self.sensor_type.append(1)

                for t in range(duragel_start, duragel_end+1):
                    png_path = tacquad_outdoor_dir + item_name +'/duragel/' + str(t) +'.png'
                    self.datalist.append(png_path)
                    self.sensor_type.append(4)

        with open(obreal_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.datalist.append(obreal_dir + row[0])
                self.sensor_type.append(2)


        with open(visgel_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.datalist.append(visgel_dir + row[0])
                self.sensor_type.append(0)

        with open(yuan18_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.datalist.append(yuan18_dir + row[0])
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
        
        print(len(self.datalist), len(self.sensor_type))

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
        
        return img, self.sensor_type[index]

class PretrainDataset_Contact_video(Dataset):
    def __init__(self, mode='train'):

        self.datalist = []
        self.sensor_type = []

        with open(tacquad_indoor_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                item_name = row[0]

                gelsight_start = int(row[1])
                gelsight_end = int(row[2])

                digit_start = int(row[3])
                digit_end = int(row[4])

                duragel_start = int(row[5])
                duragel_end = int(row[6])

                for t in range(gelsight_start, gelsight_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_indoor_dir + item_name +'/gelsight/' + str(t) +'.png'
                    png_path_2 = tacquad_indoor_dir + item_name +'/gelsight/' + str(t-1) +'.png'
                    png_path_1 = tacquad_indoor_dir + item_name +'/gelsight/' + str(t-2) +'.png'
                    png_path_0 = tacquad_indoor_dir + item_name +'/gelsight/' + str(t-3) +'.png'
                    self.datalist.append([png_path_0, png_path_1, png_path_2, png_path_3])
                    self.sensor_type.append(3)
                
                for t in range(digit_start, digit_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_indoor_dir + item_name +'/digit/' + str(t) +'.png'
                    png_path_2 = tacquad_indoor_dir + item_name +'/digit/' + str(t-1) +'.png'
                    png_path_1 = tacquad_indoor_dir + item_name +'/digit/' + str(t-2) +'.png'
                    png_path_0 = tacquad_indoor_dir + item_name +'/digit/' + str(t-3) +'.png'
                    self.datalist.append([png_path_0, png_path_1, png_path_2, png_path_3])
                    self.sensor_type.append(1)

                for t in range(duragel_start, duragel_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_indoor_dir + item_name +'/duragel/' + str(t) +'.png'
                    png_path_2 = tacquad_indoor_dir + item_name +'/duragel/' + str(t-1) +'.png'
                    png_path_1 = tacquad_indoor_dir + item_name +'/duragel/' + str(t-2) +'.png'
                    png_path_0 = tacquad_indoor_dir + item_name +'/duragel/' + str(t-3) +'.png'
                    self.datalist.append([png_path_0, png_path_1, png_path_2, png_path_3])
                    self.sensor_type.append(4)

        with open(tacquad_outdoor_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                item_name = row[0]

                gelsight_start = int(row[1])
                gelsight_end = int(row[2])

                digit_start = int(row[3])
                digit_end = int(row[4])

                duragel_start = int(row[5])
                duragel_end = int(row[6])

                for t in range(gelsight_start, gelsight_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_outdoor_dir + item_name +'/gelsight/' + str(t) +'.png'
                    png_path_2 = tacquad_outdoor_dir + item_name +'/gelsight/' + str(t-1) +'.png'
                    png_path_1 = tacquad_outdoor_dir + item_name +'/gelsight/' + str(t-2) +'.png'
                    png_path_0 = tacquad_outdoor_dir + item_name +'/gelsight/' + str(t-3) +'.png'
                    self.datalist.append([png_path_0, png_path_1, png_path_2, png_path_3])
                    self.sensor_type.append(3)
                
                for t in range(digit_start, digit_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_outdoor_dir + item_name +'/digit/' + str(t) +'.png'
                    png_path_2 = tacquad_outdoor_dir + item_name +'/digit/' + str(t-1) +'.png'
                    png_path_1 = tacquad_outdoor_dir + item_name +'/digit/' + str(t-2) +'.png'
                    png_path_0 = tacquad_outdoor_dir + item_name +'/digit/' + str(t-3) +'.png'
                    self.datalist.append([png_path_0, png_path_1, png_path_2, png_path_3])
                    self.sensor_type.append(1)

                for t in range(duragel_start, duragel_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_outdoor_dir + item_name +'/duragel/' + str(t) +'.png'
                    png_path_2 = tacquad_outdoor_dir + item_name +'/duragel/' + str(t-1) +'.png'
                    png_path_1 = tacquad_outdoor_dir + item_name +'/duragel/' + str(t-2) +'.png'
                    png_path_0 = tacquad_outdoor_dir + item_name +'/duragel/' + str(t-3) +'.png'
                    self.datalist.append([png_path_0, png_path_1, png_path_2, png_path_3])
                    self.sensor_type.append(4)

        with open(obreal_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                image_id = int(row[0].split('gelsight/')[1].split('.')[0])
                if image_id >= 3:
                    image_0 = str(image_id - 3) +'.png'
                    image_1 = str(image_id - 2) +'.png'
                    image_2 = str(image_id - 1) +'.png'
                    now_folder = obreal_dir + row[0].split('gelsight/')[0] + 'gelsight/'
                    self.datalist.append([now_folder + image_0, now_folder + image_1, now_folder + image_2, row[0]])
                    self.sensor_type.append(2)


        with open(visgel_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                image_id = int(row[0].split('/frame')[1].split('.')[0])
                if image_id >= 3:
                    image_0 = 'frame' + str(image_id - 3).zfill(4) +'.jpg'
                    image_1 = 'frame' + str(image_id - 2).zfill(4) +'.jpg'
                    image_2 = 'frame' + str(image_id - 1).zfill(4) +'.jpg'
                    now_folder = visgel_dir + row[0].split('/frame')[0] + '/'
                    self.datalist.append([now_folder + image_0, now_folder + image_1, now_folder + image_2, row[0]])
                    self.sensor_type.append(0)

        with open(yuan18_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                image_id = int(row[0].split('gelsight_frame/')[1].split('.')[0])
                if image_id >= 3:
                    image_0 = str(image_id - 3).zfill(4) +'.png'
                    image_1 = str(image_id - 2).zfill(4) +'.png'
                    image_2 = str(image_id - 1).zfill(4) +'.png'
                    now_folder = yuan18_dir + row[0].split('gelsight_frame/')[0] + 'gelsight_frame/'
                    self.datalist.append([now_folder + image_0, now_folder + image_1, now_folder + image_2, row[0]])
                    self.sensor_type.append(0)

        with open(TAG_file,'r') as file:
            for row in file:
                image = row.split(',')[0]
                folder = image.split('/')[0]
                image_name = image.split('/')[1]
                now_folder = TAG_dir + folder + '/gelsight_frame/'
                image_id = int(image_name.split('.')[0])
                if image_id >= 3:
                    image_0 = str(image_id - 3).zfill(10) +'.jpg'
                    image_1 = str(image_id - 2).zfill(10) +'.jpg'
                    image_2 = str(image_id - 1).zfill(10) +'.jpg'

                    self.datalist.append([now_folder + image_0, now_folder + image_1, now_folder + image_2, now_folder + image_name])
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
                        if file.startswith(str(image_id-3)):
                            image_0 = file
                        elif file.startswith(str(image_id-2)):
                            image_1 = file
                        elif file.startswith(str(image_id-1)):
                            image_2 = file
                        
                        if image_0 and image_1 and image_2:
                            break

                    if image_0 and image_1 and image_2:
                        now_image_folder = now_data_folder + image.split('/')[0]+'/tactile/'
                        if os.path.exists(now_image_folder + image_0) and os.path.exists(now_data_folder + image):
                            self.datalist.append([now_image_folder + image_0, now_image_folder + image_1, now_image_folder + image_2, now_data_folder + image])
                            self.sensor_type.append(1)


        for folder in os.listdir(ycb_dir):
            now_folder = ycb_dir + folder + '/'
            if os.path.isdir(now_folder):
                for now_data in ['dataset_0','dataset_1','dataset_2','dataset_3','dataset_4']:
                    now_image_folder = now_folder + now_data + '/frames/'
                    for image in os.listdir(now_image_folder):
                        image_id = int(image.split('_')[1].split('.')[0])
                        if image_id >= 9:
                            image_0 = 'frame_' + str(image_id - 9).zfill(7) +'.jpg'
                            image_1 = 'frame_' + str(image_id - 6).zfill(7) +'.jpg'
                            image_2 = 'frame_' + str(image_id - 3).zfill(7) +'.jpg'
                            if os.path.exists(now_image_folder + image_0) and os.path.exists(now_image_folder + image):
                                self.datalist.append([now_image_folder + image_0, now_image_folder + image_1, now_image_folder + image_2, now_image_folder + image])
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
                    image_0 = str(image_id - 3).zfill(10) +'.jpg'
                    image_1 = str(image_id - 2).zfill(10) +'.jpg'
                    image_2 = str(image_id - 1).zfill(10) +'.jpg'
                    self.datalist.append([now_folder + image_0, now_folder + image_1, now_folder + image_2, now_folder + image_name])
                    self.sensor_type.append(3)

        print(len(self.datalist), len(self.sensor_type))

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
        return len(self.datalist)

    def __getitem__(self, index):

        img0 = Image.open(self.datalist[index][0]).convert('RGB')
        img1 = Image.open(self.datalist[index][1]).convert('RGB')
        img2 = Image.open(self.datalist[index][2]).convert('RGB')
        img3 = Image.open(self.datalist[index][3]).convert('RGB')

        img0 = self.to_tensor(img0).unsqueeze(0)
        img1 = self.to_tensor(img1).unsqueeze(0)
        img2 = self.to_tensor(img2).unsqueeze(0)
        img3 = self.to_tensor(img3).unsqueeze(0)
        img = torch.cat([img0, img1, img2, img3])
        img = self.transform(img)
        
        return img, self.sensor_type[index]
