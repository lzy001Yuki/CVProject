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

SSVTP_dir = 'tactile_datasets/TVL/tvl_dataset/ssvtp/'
TAG_dir = '/inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/zhiyan/Touch_and_Go/touch_and_go/dataset'
obreal_dir = 'tactile_datasets/objectfolder/real/tactile/'
visgel_dir = 'tactile_datasets/visgel/images/touch/'
yuan18_dir = 'tactile_datasets/yuan18/Data_ICRA18/Data/'
TVL_dir = 'tactile_datasets/TVL/tvl_dataset/hct/'
ycb_dir = 'tactile_datasets/YCB-Slide/real/'
octopi_dir = 'tactile_datasets/octopi/'
text_dir = 'tactile_datasets/text/'

TAG_file = 'tactile_datasets/contact_text_tag_notest.csv'
obreal_file = 'tactile_datasets/contact_text_obj.csv'
visgel_file = 'tactile_datasets/contact_visgel.csv'
yuan18_file = 'tactile_datasets/contact_yuan.csv'
octopi_file = 'tactile_datasets/contact_text_octopi.csv'
TVL_file = 'tactile_datasets/contact_text_tvl.csv'

tacquad_indoor_dir = 'tactile_datasets/tacquad/data_indoor/'
tacquad_outdoor_dir = 'tactile_datasets/tacquad/data_outdoor/'

tacquad_indoor_file = 'tactile_datasets/tacquad/contact_indoor.csv'
tacquad_outdoor_file = 'tactile_datasets/tacquad/contact_outdoor.csv'

tacquad_text_dir = 'tactile_datasets/text_tacquad/'

class PretrainDataset_integrate(Dataset):
    def __init__(self, args, mode='train'):

        self.datalist = []
        self.visionlist = []
        self.textlist = []
        self.sensor_type = []
        # gelsight 0
        # digit 1
        # gelslim 2
        # gelsight mini 3
        # duragel 4

        # with open(obreal_file,'r') as file:
        #     csv_reader = csv.reader(file)
        #     for row in csv_reader:
        #         self.datalist.append(obreal_dir + row[0])
        #         self.visionlist.append(obreal_dir + row[1])
        #         self.textlist.append(text_dir + 'obj_' + row[2] +'.pt')
        #         self.sensor_type.append(2)

        with open(TAG_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                folder = row[0]
                image_id = row[1]
                test_flag = int(row[3])

                # A simple resampling method to create more text-vision-touch triplets for GelSight sensor
                for tt in range(2):
                    if test_flag == 1:
                        self.textlist.append(-1)
                    else:
                        self.textlist.append(text_dir + 'tag_' + row[2] +'.pt')
                    self.visionlist.append(TAG_dir + folder + '/video_frame/' + image_id)
                    self.datalist.append(TAG_dir + folder + '/gelsight_frame/' + image_id)
                    self.sensor_type.append(0)

        # for item in os.listdir(SSVTP_dir+'/images_tac/'):
        #     image_id = item.split('_')[1]
        #     tactile_path = SSVTP_dir+'/images_tac/'+item
        #     image_path = SSVTP_dir+'/images_rgb/'+item.replace('tac', 'rgb')
        #     self.textlist.append(text_dir + 'ssvtp_' + image_id +'.pt')
        #     self.datalist.append(tactile_path)
        #     self.visionlist.append(image_path)
        #     self.sensor_type.append(1)

        # with open(TVL_file,'r') as file:
        #     csv_reader = csv.reader(file)
        #     for row in csv_reader:
        #         image_id = row[0]
        #         self.textlist.append(text_dir + 'tvl_' + row[1] +'.pt')
        #         self.visionlist.append(TVL_dir + image_id.replace('tactile', 'vision'))
        #         self.datalist.append(TVL_dir + image_id)
        #         self.sensor_type.append(1)

        # with open(visgel_file,'r') as file:
        #     csv_reader = csv.reader(file)
        #     for row in csv_reader:
        #         self.datalist.append(visgel_dir + row[0])
        #         self.visionlist.append(visgel_dir.replace('touch', 'vision') + row[0])
        #         self.sensor_type.append(0)
        #         self.textlist.append(-1)
        
        # with open(octopi_file,'r') as file:
        #     csv_reader = csv.reader(file)
        #     for row in csv_reader:
        #         # A simple resampling method to create more samples for GelSight Mini sensor
        #         for tt in range(3):
        #             self.datalist.append(octopi_dir+row[0])
        #             self.textlist.append(text_dir+'octopi_'+row[1]+'.pt')
        #             self.visionlist.append(-1)
        #             self.sensor_type.append(3)

        # with open(tacquad_indoor_file,'r') as file:
        #     csv_reader = csv.reader(file)
        #     now_id = 0
        #     for row in csv_reader:
        #         item_name = row[0]

        #         gelsight_start = int(row[1])
        #         gelsight_end = int(row[2])

        #         digit_start = int(row[3])
        #         digit_end = int(row[4])

        #         duragel_start = int(row[5])
        #         duragel_end = int(row[6])

        #         for t in range(gelsight_start, gelsight_end+1):

        #             png_path = tacquad_indoor_dir + item_name +'/gelsight/' + str(t) +'.png'

        #             len_touch = len(os.listdir(tacquad_indoor_dir + item_name +'/gelsight/'))
        #             len_image = len(os.listdir(tacquad_indoor_dir + item_name +'/img_gelsight/'))
        #             time_point = t / (len_touch*1.0)
        #             vision_id = int(time_point * len_image)
                    
        #             image_path = tacquad_indoor_dir + item_name +'/img_gelsight/' + str(vision_id) +'.png'

        #             self.visionlist.append(image_path)
        #             self.datalist.append(png_path)
        #             self.textlist.append(tacquad_text_dir + 'tacquad_indoor_' + str(now_id) +'.pt')
        #             self.sensor_type.append(3)
                
        #         for t in range(digit_start, digit_end+1):

        #             png_path = tacquad_indoor_dir + item_name +'/digit/' + str(t) +'.png'

        #             len_touch = len(os.listdir(tacquad_indoor_dir + item_name +'/digit/'))
        #             len_image = len(os.listdir(tacquad_indoor_dir + item_name +'/img_digit/'))
        #             time_point = t / (len_touch*1.0)
        #             vision_id = int(time_point * len_image)
                    
        #             image_path = tacquad_indoor_dir + item_name +'/img_digit/' + str(vision_id) +'.png'

        #             self.visionlist.append(image_path)
        #             self.datalist.append(png_path)
        #             self.textlist.append(tacquad_text_dir + 'tacquad_indoor_' + str(now_id) +'.pt')
        #             self.sensor_type.append(1)

        #         for t in range(duragel_start, duragel_end+1):

        #             png_path = tacquad_indoor_dir + item_name +'/duragel/' + str(t) +'.png'

        #             len_touch = len(os.listdir(tacquad_indoor_dir + item_name +'/duragel/'))
        #             len_image = len(os.listdir(tacquad_indoor_dir + item_name +'/img_duragel/'))
        #             time_point = t / (len_touch*1.0)
        #             vision_id = int(time_point * len_image)
                    
        #             image_path = tacquad_indoor_dir + item_name +'/img_duragel/' + str(vision_id) +'.png'

        #             self.visionlist.append(image_path)
        #             self.datalist.append(png_path)
        #             self.textlist.append(tacquad_text_dir + 'tacquad_indoor_' + str(now_id) +'.pt')
        #             self.sensor_type.append(4)
                
        #         now_id += 1

        # with open(tacquad_outdoor_file,'r') as file:
        #     csv_reader = csv.reader(file)
        #     now_id = 0
        #     for row in csv_reader:
        #         item_name = row[0]

        #         gelsight_start = int(row[1])
        #         gelsight_end = int(row[2])

        #         digit_start = int(row[3])
        #         digit_end = int(row[4])

        #         duragel_start = int(row[5])
        #         duragel_end = int(row[6])

        #         for t in range(gelsight_start, gelsight_end+1):

        #             png_path = tacquad_outdoor_dir + item_name +'/gelsight/' + str(t) +'.png'

        #             len_touch = len(os.listdir(tacquad_outdoor_dir + item_name +'/gelsight/'))
        #             len_image = len(os.listdir(tacquad_outdoor_dir + item_name +'/img_gelsight/'))
        #             time_point = t / (len_touch*1.0)
        #             vision_id = int(time_point * len_image)
                    
        #             image_path = tacquad_outdoor_dir + item_name +'/img_gelsight/' + str(vision_id) +'.png'

        #             self.visionlist.append(image_path)
        #             self.datalist.append(png_path)
        #             self.textlist.append(tacquad_text_dir + 'tacquad_outdoor_' + str(now_id) +'.pt')
        #             self.sensor_type.append(3)
                
        #         for t in range(digit_start, digit_end+1):

        #             png_path = tacquad_outdoor_dir + item_name +'/digit/' + str(t) +'.png'

        #             len_touch = len(os.listdir(tacquad_outdoor_dir + item_name +'/digit/'))
        #             len_image = len(os.listdir(tacquad_outdoor_dir + item_name +'/img_digit/'))
        #             time_point = t / (len_touch*1.0)
        #             vision_id = int(time_point * len_image)
                    
        #             image_path = tacquad_outdoor_dir + item_name +'/img_digit/' + str(vision_id) +'.png'

        #             self.visionlist.append(image_path)
        #             self.datalist.append(png_path)
        #             self.textlist.append(tacquad_text_dir + 'tacquad_outdoor_' + str(now_id) +'.pt')
        #             self.sensor_type.append(1)

        #         for t in range(duragel_start, duragel_end+1):

        #             png_path = tacquad_outdoor_dir + item_name +'/duragel/' + str(t) +'.png'

        #             len_touch = len(os.listdir(tacquad_outdoor_dir + item_name +'/duragel/'))
        #             len_image = len(os.listdir(tacquad_outdoor_dir + item_name +'/img_duragel/'))
        #             time_point = t / (len_touch*1.0)
        #             vision_id = int(time_point * len_image)
                    
        #             image_path = tacquad_outdoor_dir + item_name +'/img_duragel/' + str(vision_id) +'.png'

        #             self.visionlist.append(image_path)
        #             self.datalist.append(png_path)
        #             self.textlist.append(tacquad_text_dir + 'tacquad_outdoor_' + str(now_id) +'.pt')
        #             self.sensor_type.append(4)

        #         now_id += 1

        if not args.no_mae:
            with open(yuan18_file,'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    self.datalist.append(yuan18_dir + row[0])
                    self.sensor_type.append(0)
                    self.visionlist.append(-1)
                    self.textlist.append(-1)


            for folder in os.listdir(ycb_dir):
                now_folder = ycb_dir + folder + '/'
                if os.path.isdir(now_folder):
                    for now_data in ['dataset_0','dataset_1','dataset_2','dataset_3','dataset_4']:
                        now_image_folder = now_folder + now_data + '/frames/'
                        for image in os.listdir(now_image_folder):
                            self.datalist.append(now_image_folder + image)
                            self.sensor_type.append(1)
                            self.visionlist.append(-1)
                            self.textlist.append(-1)
        
        print(len(self.datalist), len(self.textlist), len(self.visionlist))

        if mode == 'train':
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

            self.vision_transform = transforms.Compose([  
                    transforms.RandomRotation(degrees=(-20, 20)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop((224,224)),
                    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

            self.vision_transform = transforms.Compose([  
                    transforms.ToTensor(),
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop((224,224)),
                    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
                ])

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        touch = Image.open(self.datalist[index]).convert('RGB')
        
        touch = self.transform(touch)

        vision_flag = 0
        text_flag = 0

        if self.visionlist[index] != -1:
            vision = Image.open(self.visionlist[index]).convert('RGB')
            vision = self.vision_transform(vision)
            vision_flag = 1
        else:
            vision = torch.zeros((3, 224, 224)).float()

        if self.textlist[index] != -1:
            text, mask = torch.load(self.textlist[index])
            text = text.int()
            mask = mask.int()
            text_flag = 1
        else:
            text = torch.zeros(77).int()
            mask = torch.zeros(77).int()
        
        return touch, vision, text, mask, self.sensor_type[index], vision_flag, text_flag


class PretrainDataset_video_integrate(Dataset):
    def __init__(self, args, mode='train'):

        self.datalist = []
        self.visionlist = []
        self.textlist = []
        self.sensor_type = []
        # gelsight 0
        # digit 1
        # gelslim 2
        # gelsight mini 3
        # duragel 4

        with open(obreal_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                #self.datalist.append(obreal_dir + row[0])
                image_id = int(row[0].split('gelsight/')[1].split('.')[0])
                if image_id >= 3:
                    image_0 = str(image_id - 3) +'.png'
                    image_1 = str(image_id - 2) +'.png'
                    image_2 = str(image_id - 1) +'.png'
                    now_folder = row[0].split('gelsight/')[0] + 'gelsight/'
                    self.datalist.append([obreal_dir + now_folder + image_0, obreal_dir + now_folder + image_1, obreal_dir + now_folder + image_2, obreal_dir + row[0]])
                    self.visionlist.append(obreal_dir + row[1])
                    self.textlist.append(text_dir + 'obj_' + row[2] +'.pt')
                    self.sensor_type.append(2)

        with open(TAG_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                folder = row[0]
                image_name = row[1]
                image_id = int(image_name.split('.')[0])
                test_flag = int(row[3])
                if image_id >= 3:
                    image_0 = str(image_id - 3).zfill(10) +'.jpg'
                    image_1 = str(image_id - 2).zfill(10) +'.jpg'
                    image_2 = str(image_id - 1).zfill(10) +'.jpg'
                    # A simple resampling method to create more text-vision-touch triplets for GelSight sensor
                    for tt in range(2):
                        self.datalist.append([TAG_dir + folder + '/gelsight_frame/' + image_0, TAG_dir + folder + '/gelsight_frame/' + image_1, TAG_dir + folder + '/gelsight_frame/' + image_2, TAG_dir + folder + '/gelsight_frame/' + image_name])
                        if test_flag == 1:
                            self.textlist.append(-1)
                        else:
                            self.textlist.append(text_dir + 'tag_' + row[2] +'.pt')
                        self.visionlist.append(TAG_dir + folder + '/video_frame/' + image_name)
                        self.sensor_type.append(0)


        with open(TVL_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                image_name = row[0]
                image_id = int(image_name.split('/tactile/')[1].split('-')[0])
                image_list = os.listdir(TVL_dir + image_name.split('/tactile/')[0]+'/tactile/')
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
                    now_image_folder = TVL_dir + image_name.split('/tactile/')[0]+'/tactile/'
                    self.datalist.append([now_image_folder + image_0, now_image_folder + image_1, now_image_folder + image_2, TVL_dir + image_name])
                    self.textlist.append(text_dir + 'tvl_' + row[1] +'.pt')
                    self.visionlist.append(TVL_dir + image_name.replace('tactile', 'vision'))
                    #self.datalist.append(TVL_dir + image_name)
                    self.sensor_type.append(1)

        with open(octopi_file,'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                image = row[0]
                folder = image.split('/')[1]
                image_name = image.split('/')[2]

                now_folder = octopi_dir + 'processed/' + folder + '/'

                image_id = int(image_name.split('.')[0])
                if image_id >= 3:
                    # A simple resampling method to create more samples for GelSight Mini sensor
                    for tt in range(3):
                        image_0 = str(image_id - 3).zfill(10) +'.jpg'
                        image_1 = str(image_id - 2).zfill(10) +'.jpg'
                        image_2 = str(image_id - 1).zfill(10) +'.jpg'
                        self.datalist.append([now_folder + image_0, now_folder + image_1, now_folder + image_2, now_folder + image_name])
                        self.textlist.append(text_dir+'octopi_'+row[1]+'.pt')
                        self.visionlist.append(-1)
                        self.sensor_type.append(3)

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
                    self.visionlist.append(visgel_dir.replace('touch', 'vision') + row[0])
                    self.textlist.append(-1)
                    self.sensor_type.append(0)
        
        with open(tacquad_indoor_file,'r') as file:
            csv_reader = csv.reader(file)
            now_id = 0
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

                    len_touch = len(os.listdir(tacquad_indoor_dir + item_name +'/gelsight/'))
                    len_image = len(os.listdir(tacquad_indoor_dir + item_name +'/img_gelsight/'))
                    time_point = t / (len_touch*1.0)
                    vision_id = int(time_point * len_image)
                    
                    image_path = tacquad_indoor_dir + item_name +'/img_gelsight/' + str(vision_id) +'.png'

                    self.visionlist.append(image_path)
                    self.datalist.append([png_path_0, png_path_1, png_path_2, png_path_3])
                    self.textlist.append(tacquad_text_dir + 'tacquad_indoor_' + str(now_id) +'.pt')
                    self.sensor_type.append(3)
                
                for t in range(digit_start, digit_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_indoor_dir + item_name +'/digit/' + str(t) +'.png'
                    png_path_2 = tacquad_indoor_dir + item_name +'/digit/' + str(t-1) +'.png'
                    png_path_1 = tacquad_indoor_dir + item_name +'/digit/' + str(t-2) +'.png'
                    png_path_0 = tacquad_indoor_dir + item_name +'/digit/' + str(t-3) +'.png'

                    len_touch = len(os.listdir(tacquad_indoor_dir + item_name +'/digit/'))
                    len_image = len(os.listdir(tacquad_indoor_dir + item_name +'/img_digit/'))
                    time_point = t / (len_touch*1.0)
                    vision_id = int(time_point * len_image)
                    
                    image_path = tacquad_indoor_dir + item_name +'/img_digit/' + str(vision_id) +'.png'

                    self.visionlist.append(image_path)
                    self.datalist.append([png_path_0, png_path_1, png_path_2, png_path_3])
                    self.textlist.append(tacquad_text_dir + 'tacquad_indoor_' + str(now_id) +'.pt')
                    self.sensor_type.append(1)

                for t in range(duragel_start, duragel_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_indoor_dir + item_name +'/duragel/' + str(t) +'.png'
                    png_path_2 = tacquad_indoor_dir + item_name +'/duragel/' + str(t-1) +'.png'
                    png_path_1 = tacquad_indoor_dir + item_name +'/duragel/' + str(t-2) +'.png'
                    png_path_0 = tacquad_indoor_dir + item_name +'/duragel/' + str(t-3) +'.png'

                    len_touch = len(os.listdir(tacquad_indoor_dir + item_name +'/duragel/'))
                    len_image = len(os.listdir(tacquad_indoor_dir + item_name +'/img_duragel/'))
                    time_point = t / (len_touch*1.0)
                    vision_id = int(time_point * len_image)
                    
                    image_path = tacquad_indoor_dir + item_name +'/img_duragel/' + str(vision_id) +'.png'

                    self.visionlist.append(image_path)
                    self.datalist.append([png_path_0, png_path_1, png_path_2, png_path_3])
                    self.textlist.append(tacquad_text_dir + 'tacquad_indoor_' + str(now_id) +'.pt')
                    self.sensor_type.append(4)
                
                now_id += 1

        with open(tacquad_outdoor_file,'r') as file:
            csv_reader = csv.reader(file)
            now_id = 0
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

                    len_touch = len(os.listdir(tacquad_outdoor_dir + item_name +'/gelsight/'))
                    len_image = len(os.listdir(tacquad_outdoor_dir + item_name +'/img_gelsight/'))
                    time_point = t / (len_touch*1.0)
                    vision_id = int(time_point * len_image)
                    
                    image_path = tacquad_outdoor_dir + item_name +'/img_gelsight/' + str(vision_id) +'.png'

                    self.visionlist.append(image_path)
                    self.datalist.append([png_path_0, png_path_1, png_path_2, png_path_3])
                    self.textlist.append(tacquad_text_dir + 'tacquad_outdoor_' + str(now_id) +'.pt')
                    self.sensor_type.append(3)
                
                for t in range(digit_start, digit_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_outdoor_dir + item_name +'/digit/' + str(t) +'.png'
                    png_path_2 = tacquad_outdoor_dir + item_name +'/digit/' + str(t-1) +'.png'
                    png_path_1 = tacquad_outdoor_dir + item_name +'/digit/' + str(t-2) +'.png'
                    png_path_0 = tacquad_outdoor_dir + item_name +'/digit/' + str(t-3) +'.png'

                    len_touch = len(os.listdir(tacquad_outdoor_dir + item_name +'/digit/'))
                    len_image = len(os.listdir(tacquad_outdoor_dir + item_name +'/img_digit/'))
                    time_point = t / (len_touch*1.0)
                    vision_id = int(time_point * len_image)
                    
                    image_path = tacquad_outdoor_dir + item_name +'/img_digit/' + str(vision_id) +'.png'

                    self.visionlist.append(image_path)
                    self.datalist.append([png_path_0, png_path_1, png_path_2, png_path_3])
                    self.textlist.append(tacquad_text_dir + 'tacquad_outdoor_' + str(now_id) +'.pt')
                    self.sensor_type.append(1)

                for t in range(duragel_start, duragel_end+1):
                    if t<3:
                        continue
                    png_path_3 = tacquad_outdoor_dir + item_name +'/duragel/' + str(t) +'.png'
                    png_path_2 = tacquad_outdoor_dir + item_name +'/duragel/' + str(t-1) +'.png'
                    png_path_1 = tacquad_outdoor_dir + item_name +'/duragel/' + str(t-2) +'.png'
                    png_path_0 = tacquad_outdoor_dir + item_name +'/duragel/' + str(t-3) +'.png'

                    len_touch = len(os.listdir(tacquad_outdoor_dir + item_name +'/duragel/'))
                    len_image = len(os.listdir(tacquad_outdoor_dir + item_name +'/img_duragel/'))
                    time_point = t / (len_touch*1.0)
                    vision_id = int(time_point * len_image)
                    
                    image_path = tacquad_outdoor_dir + item_name +'/img_duragel/' + str(vision_id) +'.png'

                    self.visionlist.append(image_path)
                    self.datalist.append([png_path_0, png_path_1, png_path_2, png_path_3])
                    self.textlist.append(tacquad_text_dir + 'tacquad_outdoor_' + str(now_id) +'.pt')
                    self.sensor_type.append(4)

                now_id += 1

        if not args.no_mae:
            with open(yuan18_file,'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    image_id = int(row[0].split('gelsight_frame/')[1].split('.')[0])
                    if image_id >= 3:
                        self.textlist.append(-1)
                        self.visionlist.append(-1)
                        image_0 = str(image_id - 3).zfill(4) +'.png'
                        image_1 = str(image_id - 2).zfill(4) +'.png'
                        image_2 = str(image_id - 1).zfill(4) +'.png'
                        now_folder = yuan18_dir + row[0].split('gelsight_frame/')[0] + 'gelsight_frame/'
                        self.datalist.append([now_folder + image_0, now_folder + image_1, now_folder + image_2, row[0]])
                        self.sensor_type.append(0)


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
                                    self.textlist.append(-1)
                                    self.visionlist.append(-1)
                                    self.sensor_type.append(1)

        print(len(self.datalist), len(self.textlist), len(self.visionlist))

        if mode == 'train':
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224), antialias=True),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.3),
                    #transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

            self.vision_transform = transforms.Compose([  
                    transforms.RandomRotation(degrees=(-20, 20)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop((224,224)),
                    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
                ])
        else:
            self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224), antialias=True),
                    #transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

            self.vision_transform = transforms.Compose([  
                    transforms.ToTensor(),
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop((224,224)),
                    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
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
        touch = self.transform(img)

        vision_flag = 0
        text_flag = 0

        if self.visionlist[index] != -1:
            vision = Image.open(self.visionlist[index]).convert('RGB')
            vision = self.vision_transform(vision)
            vision_flag = 1
        else:
            vision = torch.zeros((3, 224, 224)).float()

        if self.textlist[index] != -1:
            text, mask = torch.load(self.textlist[index])
            text = text.int()
            mask = mask.int()
            text_flag = 1
        else:
            text = torch.zeros(77).int()
            mask = torch.zeros(77).int()
        
        return touch, vision, text, mask, self.sensor_type[index], vision_flag, text_flag
