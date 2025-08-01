# This file contains all the helper functions used for the purpose of this project
# 

# Import necessary libraries
import os
import json
import argparse
from tqdm import tqdm
import csv

import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree
from PIL import Image
import glob, shutil



# This function converts annotations from BDD to COCO format
def bdd2coco_detection(id_dict, labeled_images, fn):

    images = list()
    annotations = list()

    counter = 0
    for i in tqdm(labeled_images):
        counter += 1
        image = dict()
        image['file_name'] = i['name']
        image['height'] = 720
        image['width'] = 1280

        image['id'] = counter

        empty_image = True

        for label in i['labels']:
            annotation = dict()
            category=label['category']
            if (category == "traffic light"):
                color = label['attributes']['trafficLightColor']
                category = "tl_" + color
            if category in id_dict.keys():
                empty_image = False
                annotation["iscrowd"] = 0
                annotation["image_id"] = image['id']
                x1 = label['box2d']['x1']
                y1 = label['box2d']['y1']
                x2 = label['box2d']['x2']
                y2 = label['box2d']['y2']
                annotation['bbox'] = [x1, y1, x2-x1, y2-y1]
                annotation['area'] = float((x2 - x1) * (y2 - y1))
                annotation['category_id'] = id_dict[category]
                annotation['ignore'] = 0
                annotation['id'] = label['id']
                annotation['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                annotations.append(annotation)

        if empty_image:
            continue

        images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print('saving...')
    json_string = json.dumps(attr_dict)
    with open(fn, "w") as file:
        file.write(json_string)


if __name__ == '__main__':

    label_dir="/content/drive/MyDrive/CS5500/bdd_data_files/bdd100k/labels/"
    save_path="/content/drive/MyDrive/CS5500/bdd_data_files/bdd100k/labels/"

    attr_dict = dict()
    attr_dict["categories"] = [
        {"supercategory": "none", "id": 1, "name": "person"},
        {"supercategory": "none", "id": 2, "name": "rider"},
        {"supercategory": "none", "id": 3, "name": "car"},
        {"supercategory": "none", "id": 4, "name": "truck"},
        {"supercategory": "none", "id": 5, "name": "bus"},
        {"supercategory": "none", "id": 6, "name": "train"},
        {"supercategory": "none", "id": 7, "name": "motorcycle"},
        {"supercategory": "none", "id": 8, "name": "tl_green"},
        {"supercategory": "none", "id": 9, "name": "tl_red"},
        {"supercategory": "none", "id": 10, "name": "tl_yellow"},
        {"supercategory": "none", "id": 11, "name": "tl_none"},
        {"supercategory": "none", "id": 12, "name": "traffic sign"},
        {"supercategory": "none", "id": 13, "name": "train"}
    ]

    attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

    # create BDD training set detections in COCO format
    print('Loading training set...')
    with open(os.path.join(label_dir,
                           'bdd100k_labels_images_train.json')) as f:
        train_labels = json.load(f)
    print('Converting training set...')

    out_fn = os.path.join(save_path,
                          'bdd100k_labels_images_det_coco_train.json')
    bdd2coco_detection(attr_id_dict, train_labels, out_fn)

    print('Loading validation set...')
    # create BDD validation set detections in COCO format
    with open(os.path.join(label_dir,
                           'bdd100k_labels_images_val.json')) as f:
        val_labels = json.load(f)
    print('Converting validation set...')

    out_fn = os.path.join(save_path,
                          'bdd100k_labels_images_det_coco_val.json')
    bdd2coco_detection(attr_id_dict, val_labels, out_fn)
    
    
    
   
# This function prints progress bar when converting annotations   
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s|%s| %s%% (%s/%s)  %s' % (prefix, bar, percent, iteration, total, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print("\n")
        
        


class COCO:
    """
    Handler Class for COCO Format
    """

    @staticmethod
    def parse(json_path):

        try:
            json_data = json.load(open(json_path))

            images_info = json_data["images"]
            cls_info = json_data["categories"]

            data = {}

            progress_length = len(json_data["annotations"])
            progress_cnt = 0
            printProgressBar(0, progress_length, prefix='\nCOCO Parsing:'.ljust(15), suffix='Complete', length=40)

            for anno in json_data["annotations"]:

                image_id = anno["image_id"]
                cls_id = anno["category_id"]

                filename = None
                img_width = None
                img_height = None
                cls = None

                for info in images_info:
                        if info["id"] == image_id:
                            filename, img_width, img_height = \
                                info["file_name"].split(".")[0], info["width"], info["height"]

                for category in cls_info:
                    if category["id"] == cls_id:
                        cls = category["name"]

                size = {
                    "width": img_width,
                    "height": img_height,
                    "depth": "3"
                }

                bndbox = {
                    "xmin": anno["bbox"][0],
                    "ymin": anno["bbox"][1],
                    "xmax": anno["bbox"][2] + anno["bbox"][0],
                    "ymax": anno["bbox"][3] + anno["bbox"][1]
                }

                obj_info = {
                    "name": cls,
                    "bndbox": bndbox
                }

                if filename in data:
                    obj_idx = str(int(data[filename]["objects"]["num_obj"]))
                    data[filename]["objects"][str(obj_idx)] = obj_info
                    data[filename]["objects"]["num_obj"] = int(obj_idx) + 1

                elif filename not in data:

                    obj = {
                        "num_obj": "1",
                        "0": obj_info
                    }

                    data[filename] = {
                        "size": size,
                        "objects": obj
                    }

                printProgressBar(progress_cnt + 1, progress_length, prefix='COCO Parsing:'.ljust(15), suffix='Complete', length=40)
                progress_cnt += 1

            #print(json.dumps(data, indent=4, sort_keys = True))
            return True, data

        except Exception as e:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            msg = "ERROR : {}, moreInfo : {}\t{}\t{}".format(e, exc_type, fname, exc_tb.tb_lineno)

            return False, msg
        
        



class YOLO:
    """
    Handler Class for UDACITY Format
    """

    def __init__(self, cls_list_path):
        with open(cls_list_path, 'r') as file:
            l = file.read().splitlines()

        self.cls_list = l


    def coordinateCvt2YOLO(self,size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]

        # (xmin + xmax / 2)
        x = (box[0] + box[1]) / 2.0
        # (ymin + ymax / 2)
        y = (box[2] + box[3]) / 2.0

        # (xmax - xmin) = w
        w = box[1] - box[0]
        # (ymax - ymin) = h
        h = box[3] - box[2]

        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (round(x,3), round(y,3), round(w,3), round(h,3))

    def parse(self, label_path, img_path, img_type=".png"):
        try:

            (dir_path, dir_names, filenames) = next(os.walk(os.path.abspath(label_path)))

            data = {}

            progress_length = len(filenames)
            progress_cnt = 0
            printProgressBar(0, progress_length, prefix='\nYOLO Parsing:'.ljust(15), suffix='Complete', length=40)

            for filename in filenames:

                txt = open(os.path.join(dir_path, filename), "r")

                filename = filename.split(".")[0]

                img = Image.open(os.path.join(img_path, "".join([filename, img_type])))
                img_width = str(img.size[0])
                img_height = str(img.size[1])
                img_depth = 3

                size = {
                    "width": img_width,
                    "height": img_height,
                    "depth": img_depth
                }

                obj = {}
                obj_cnt = 0

                for line in txt:
                    elements = line.split(" ")
                    name_id = elements[0]

                    xminAddxmax = float(elements[1]) * (2.0 * float(img_width))
                    yminAddymax = float(elements[2]) * (2.0 * float(img_height))

                    w = float(elements[3]) * float(img_width)
                    h = float(elements[4]) * float(img_height)

                    xmin = (xminAddxmax - w) / 2
                    ymin = (yminAddymax - h) / 2
                    xmax = xmin + w
                    ymax = ymin + h
  
                    bndbox = {
                        "xmin": float(xmin),
                        "ymin": float(ymin),
                        "xmax": float(xmax),
                        "ymax": float(ymax)
                    }


                    obj_info = {
                        "name": name_id,
                        "bndbox": bndbox
                    }

                    obj[str(obj_cnt)] =obj_info
                    obj_cnt += 1

                obj["num_obj"] =  obj_cnt

                data[filename] = {
                    "size": size,
                    "objects": obj
                }

                printProgressBar(progress_cnt + 1, progress_length, prefix='YOLO Parsing:'.ljust(15), suffix='Complete',
                                 length=40)
                progress_cnt += 1

            return True, data

        except Exception as e:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            msg = "ERROR : {}, moreInfo : {}\t{}\t{}".format(e, exc_type, fname, exc_tb.tb_lineno)

            return False, msg

    def generate(self, data):

        try:

            progress_length =len(data)
            progress_cnt = 0
            printProgressBar(0, progress_length, prefix='\nYOLO Generating:'.ljust(15), suffix='Complete', length=40)

            result = {}

            for key in data:
                img_width = int(data[key]["size"]["width"])
                img_height = int(data[key]["size"]["height"])

                contents = ""

                for idx in range(0, int(data[key]["objects"]["num_obj"])):

                    xmin = data[key]["objects"][str(idx)]["bndbox"]["xmin"]
                    ymin = data[key]["objects"][str(idx)]["bndbox"]["ymin"]
                    xmax = data[key]["objects"][str(idx)]["bndbox"]["xmax"]
                    ymax = data[key]["objects"][str(idx)]["bndbox"]["ymax"]

                    b = (float(xmin), float(xmax), float(ymin), float(ymax))
                    bb = self.coordinateCvt2YOLO((img_width, img_height), b)

                    cls_id = self.cls_list.index(data[key]["objects"][str(idx)]["name"])

                    bndbox = "".join(["".join([str(e), " "]) for e in bb])
                    contents = "".join([contents, str(cls_id), " ", bndbox[:-1], "\n"])

                result[key] = contents

                printProgressBar(progress_cnt + 1, progress_length, prefix='YOLO Generating:'.ljust(15),
                                 suffix='Complete',
                                 length=40)
                progress_cnt += 1

            return True, result

        except Exception as e:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            msg = "ERROR : {}, moreInfo : {}\t{}\t{}".format(e, exc_type, fname, exc_tb.tb_lineno)

            return False, msg

    def save(self, data, save_path, img_path, img_type, manipast_path):

        try:

            progress_length = len(data)
            progress_cnt = 0
            printProgressBar(0, progress_length, prefix='\nYOLO Saving:'.ljust(15), suffix='Complete', length=40)

            with open(os.path.abspath(os.path.join(manipast_path, "manifast.txt")), "w") as manipast_file:

                for key in data:
                    manipast_file.write(os.path.abspath(os.path.join(img_path, "".join([key, img_type, "\n"]))))

                    with open(os.path.abspath(os.path.join(save_path, "".join([key, ".txt"]))), "w") as output_txt_file:
                        output_txt_file.write(data[key])


                    printProgressBar(progress_cnt + 1, progress_length, prefix='YOLO Saving:'.ljust(15),
                                     suffix='Complete',
                                     length=40)
                    progress_cnt += 1

            return True, None

        except Exception as e:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            msg = "ERROR : {}, moreInfo : {}\t{}\t{}".format(e, exc_type, fname, exc_tb.tb_lineno)

            return False, msg
        
        
        

# Helper function for showing predictions in the images
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()
  
  

'''
Sometimes your image data set might not match with your label data set.
This code does the folowing
(1) Go through your image data set
(2) Search if the corresponding label file exist in the label data set. 
(3) If not, remove current image
'''


def copy_filter(label_dir,image_dir,target_dir_images,target_dir_labels):
    for image in os.listdir(image_dir):
        if image.endswith('jpg'):
            image_name = os.path.splitext(image)[0]

            # Corresponding label file name
            label_name = image_name + '.txt'
            image_path = image_dir + '/' + image_name + '.jpg'
            if os.path.isfile(label_dir + '/' + label_name) == False:
                print(" -- DELETE IMAGE [Label file not found -- ]")
                
                print(image_path)
#                 os.remove(image_path)
#             else:
                target_images=target_dir_images+ '/' + image_name + '.jpg'
                shutil.copy(image_path,target_dir_images )
                print(" --COPY IMAGE "+target_images)


    for label in os.listdir(label_dir):
        if label.endswith('.txt'):
            label_name = os.path.splitext(label)[0]

            # Corresponding label file name
            image_name = label_name + '.jpg'
            label_path = label_dir + '/' + label_name + '.txt'
            if os.path.isfile(image_dir + '/' + image_name) == False:
                print(" -- DELETE LABEL [Image file not found -- ]")
                print(label_path)
#                 os.remove(label_path)
#             else:
                target_labels=target_dir_labels+ '/' + label_name + '.txt'
                shutil.copy(label_path,target_labels )
                print(" --COPY lABELS "+target_labels)