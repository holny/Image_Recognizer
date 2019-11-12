#coding:utf-8
import tensorflow as tf
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import time

IMAGES_DIR = "./data/train/gen_images"
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 600
CHARS_MAX_NUM = 15  ## 15, 单个label中最大字符数

TRAIN_DATA_PATH = "./data/tfrecord/TrainsetTFrecord"  ## 训练集地址
TRAIN_DATA_TOTAL_NUM = 1706 ## Train set数据总数

TEST_DATA_PATH = "./data/tfrecord/TestsetTFrecord"
TEST_DATA_TOTAL_NUM = 0

def _get_filenames(images_dir):
    image_file_names = []
    label_file_name = ""
    for filename in os.listdir(images_dir):
        if filename.endswith(".png"):
            image_file_names.append(join(images_dir, filename))
        elif filename.endswith(".txt"):
            label_file_name = join(images_dir, filename)

    return label_file_name

def _parse_label_file(label_file_name):
    try:
        if os.path.isfile(label_file_name):
            infile = open(label_file_name, "r")
        else:
            print("并没有这个文件")
    except IOError:
        print("输入文件路径错误！")
    list = infile.readlines()
    infile.close()
    total_data = []

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
        # image = plt.imread(path)
        # # print("image.shape=",image.shape)
        # # plt.imshow(image)
        # image = _process_image(image)
        img = Image.open(path)
        ## image process
        ## 调整统一高度
        old_size = img.size     ## (w,h)
        old_width = old_size[0]
        img = img.resize((old_width,IMAGE_HEIGHT),Image.ANTIALIAS)
        img = np.array(img)
        ## 调整填充宽度
        if old_width<IMAGE_WIDTH:
            pad_width = IMAGE_WIDTH-old_width
            img = np.pad(img,((0,0),(0,pad_width),(0, 0)),"constant",constant_values=255)
        ## 保存调整后的图片
        image = Image.fromarray(img.astype('uint8')).convert('RGB')
        new_path = path.replace(".jpg",".png").replace(".jpeg",".png").replace(".png","_format.png")
        image.save(new_path)
    except IOError:
        print("Error: Image读取异常，Path:",path)
        return None
    return img

def _gen_tfrecord_from_data(path,total_data):
    data_dir = path.rsplit("/", 1)[0]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # path = path +"_%s" % (data_num)
    # time_str = time.strftime('%m%d%H%M', time.localtime(time.time()))
    # path = path + "_%s" % (time_str)
    print(path)
    writer = tf.python_io.TFRecordWriter(path)
    data_num_in_tfrecord = 0
    for i in range(len(total_data)):
        data = total_data[i]
        image_path,chars_num = data[0],data[1]
        image_path = image_path.replace("./gen_images", IMAGES_DIR, 1)
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

def main(images_dir,tfrecord_path):
    label_file_name = _get_filenames(images_dir)
    total_data = _parse_label_file(label_file_name)
    data_num = _gen_tfrecord_from_data(tfrecord_path,total_data)
    print("已生成 ",data_num," 条TFrecord数据，Image数据源:",images_dir," ,生成的TFrecord地址:",tfrecord_path)
    return data_num

if __name__ == "__main__":
    # tf_file_queue()
    data_num = main(IMAGES_DIR,TRAIN_DATA_PATH)
