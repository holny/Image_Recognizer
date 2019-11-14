#coding:utf-8
import tensorflow as tf
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import time
"""
此文件是工具文件，作用是根据label_info.txt获取label信息，然后根据每条label信息获取到每个image图片。
并把image图片和label的数据存入到Tfrecord中。

使用时，只需修改LABEL_INFO_PATH为label_info.txt所在的目录地址，支持多label_info.txt同时读取。但注意label中的image地址正确。
TF_RECORD_FILE_NAME修改为想要保存的TFrecord文件路径(是期望生成的TFrecord文件所在目录)，
TF_RECORD_FILE_NAME是期望生成的TFrecord文件名，注意"Train"是训练集的关键字，想做为训练集时TFrecord文件名要带"Train"，Test同理。
然后运行代码即可。

IMAGE_HEIGHT、IMAGE_WIDTH、CHARS_MAX_NUM需要提前设置好，
IS_RESIZE_IMAGE=True时就会在Image存入TFrecord之前裁剪尺寸(包括宽高，不裁通道)，不足的padding255，超过的resize。
并且Label存入tfrecord之前要填充到最大CHARS_MAX_NUM。label不足的pad 0，
"""
LABEL_INFO_PATH = "/Users/hly/PycharmProjects/HWR_1112/data/test_info" ## label info.txt所在的目录，支持多info.txt读取
TF_RECORD_FILE_NAME = "TestTFrecord"      ## 生成的tfrecord文件名

TF_RECORD_FILE_DIR = "./data/tfrecord/"  ## 生成的TFrecord地址

IS_RESIZE_IMAGE = True             ## 是否重新剪裁图片
IS_SAVE_AFTER_RESIZE = False        ## 剪裁后的图片是否保存在同级目录，for test。 但是剪裁后的图片还是会送往TFrecord
IMAGE_HEIGHT = 32       ## IS_RESIZE_IMAGE=True时，在存入tfrecord之前固定Image尺寸，不足padding
IMAGE_WIDTH = 64
CHARS_MAX_NUM = 1      ## 15, 单个label中最大字符数，在label存入tfrecord之前固定长度为15，不足padding

def _get_filenames(label_info_dir,key=".txt"):
    label_filename_list = []
    for filename in os.listdir(label_info_dir):
        if filename.endswith(key):
            label_file_name = join(label_info_dir, filename)
            label_filename_list.append(label_file_name)

    return label_filename_list

def _parse_label_file(label_filename_list):
    total_data = []
    for i in  range(len(label_filename_list)):
        label_file_name =  label_filename_list[i]
        print("parse_label_file--label info path=",label_file_name)
        try:
            if os.path.isfile(label_file_name):
                infile = open(label_file_name, "r")
            else:
                print("并没有这个文件,path:",label_file_name)
        except IOError:
            print("输入文件路径错误！")
        list = infile.readlines()
        infile.close()
        for i in range(len(list)):
            lineStr = list[i]
            tempList = lineStr.split(",")
            data = []
            image_path = tempList[0].replace("['","").replace("'","").strip()
            data.append(image_path)
            chars_num = len(tempList) - 1         ## 字符个数
            data.append(chars_num)
            for j in range(1,len(tempList)):
                loc = int(tempList[j].replace("[","").replace("]","").replace(r"\n","").strip())+1 ## 前移一位，留出0作CTC
                data.append(loc)
            total_data.append(data)
    print(total_data)
    print(len(total_data))
    return total_data       ## [图片路径，字符个数， 字符label]

def show_image(image,chars_num,image_label,image_shape):
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, str(chars_num)+"-"+str(image_label)+"-"+str(image_shape), ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
    plt.show()

def _get_image(path):
    try:
        img = Image.open(path)
        ## image process
        ## 调整统一高度
        if IS_RESIZE_IMAGE:
            old_size = img.size     ## (w,h)
            old_width = old_size[0]
            old_height = old_size[1]
            ## 调整图片height，不足padding，超出resize
            if old_height<IMAGE_HEIGHT:
                pad_height = IMAGE_HEIGHT - old_height
                img = np.pad(img, ((0, pad_height), (0, 0), (0, 0)), "constant", constant_values=255)
            elif old_height>IMAGE_HEIGHT:
                img = img.resize((old_width,IMAGE_HEIGHT),Image.ANTIALIAS)
            ## 调整图片width，不足padding，超出resize
            if old_width<IMAGE_WIDTH:           ## 图片宽小于规定宽就Padding
                pad_width = IMAGE_WIDTH-old_width
                img = np.pad(img,((0,0),(0,pad_width),(0, 0)),"constant",constant_values=255)
            elif old_width>IMAGE_WIDTH:         ## 图片宽大于规定宽就resize
                img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT),Image.ANTIALIAS)
            ## 保存调整后的图片
            if IS_SAVE_AFTER_RESIZE:        ## fot test 保存剪裁后的图片
                image = Image.fromarray(img.astype('uint8')).convert('RGB')
                new_path = path.replace(".jpg",".png").replace(".jpeg",".png").replace(".png","_format.png")
                image.save(new_path)
    except IOError:
        print("Error: Image读取异常，Path:",path)
        return None
    img = np.array(img)
    return img

def _gen_tfrecord_from_data(path,total_data):
    data_dir = path.rsplit("/", 1)[0]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    path = path + "_%d"%(CHARS_MAX_NUM)
    path = path +"_%s" % (len(total_data))
    time_str = time.strftime('%m%d', time.localtime(time.time()))
    path = path + "_%s" % (time_str)
    print(path)
    writer = tf.python_io.TFRecordWriter(path)
    data_num_in_tfrecord = 0
    for i in range(len(total_data)):
        data = total_data[i]
        image_path,chars_num = data[0],data[1]
        image_path = image_path.replace("./gen_images", LABEL_INFO_PATH, 1)
        image = _get_image(image_path)
        if not isinstance(image,np.ndarray):
            print("Error: 读取的图片为空！ imagePath:",image_path)
            continue
        image_shape = image.shape
        image_label = data[2:]
        ## label 填充到CHARS_MAX_NUM，padding -1
        if CHARS_MAX_NUM>len(image_label):
            label_pad_num = CHARS_MAX_NUM -len(image_label)
            pad = [0]*label_pad_num
            image_label.extend(pad)
        elif CHARS_MAX_NUM<len(image_label):
            print("Error: 长度大于",CHARS_MAX_NUM,"！ imagePath:", image_path)
            continue
        ## for test,show image
        # if i%100 ==0:
        #     show_image(image,chars_num,image_label,image_shape)

        image_raw = image.tostring()
        image_shape_raw = image_shape
        chars_num_raw = chars_num
        label_raw = np.array(image_label)
        print("gen_tfrecord--i=",i," ,image_shape=",image_shape_raw," ,chars_num=",chars_num_raw," ,label_raw=",label_raw)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                    "image_shape_raw": tf.train.Feature(int64_list=tf.train.Int64List(value=image_shape_raw)),
                    "chars_num_raw": tf.train.Feature(int64_list=tf.train.Int64List(value=[chars_num_raw])),
                    "label_raw": tf.train.Feature(int64_list=tf.train.Int64List(value=label_raw)),
                }
            )
        )
        writer.write(example.SerializeToString())
        data_num_in_tfrecord += 1
    writer.close()
    return data_num_in_tfrecord

def main(label_info_path,tfrecord_path):
    label_filename_list = _get_filenames(label_info_path,key=".txt")
    total_data = _parse_label_file(label_filename_list)
    data_num = _gen_tfrecord_from_data(tfrecord_path,total_data)
    print("已生成 ",data_num," 条TFrecord数据，label info 数据源:",label_filename_list," ,生成的TFrecord地址:",tfrecord_path)
    return data_num

if __name__ == "__main__":
    # tf_file_queue()
    data_num = main(LABEL_INFO_PATH,TF_RECORD_FILE_DIR+TF_RECORD_FILE_NAME)
