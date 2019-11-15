import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import image_to_tfrecord as data
import os
from os.path import join

# CHARS_DICT = util_data.CHARS_DICT  ## 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' ， 字典。
NUM_CLASSES = 3755 + 1  ## 为空位+1，classes最后一位(classes[26])代表空位
CHARS_MAX_NUM = data.CHARS_MAX_NUM  ## 15, 单个label中最大字符数
# IMAGE_SHAPE = util_data.IMAGE_SHAPE  ## 数据图片的shape[h,w,c]，如验证码数据集shape[64, 256, 3]

IMAGE_HEIGHT = data.IMAGE_HEIGHT
IMAGE_WIDTH = data.IMAGE_WIDTH
IMAGE_CHANNELS = 3

TEST_EPOCHS = 2
TEST_BATCH_SIZE = 100
TF_RECORD_FILE_DIR = data.TF_RECORD_FILE_DIR

TRAIN_EPOCHS = 300  ## epoch
TRAIN_BATCH_SIZE = 1024  ## batch_size


def _get_tfrecord_filenames(tfrecord_dir,key="Test"):
    label_filename_list = []
    for filename in os.listdir(tfrecord_dir):
        if filename.startswith(key) > 0:
            label_file_name = join(tfrecord_dir, filename)
            label_filename_list.append(label_file_name)
    return label_filename_list

def labels_sequence_to_sparse(labels_seq):
    """
    一个batch的labels，从sequence转换为sparse，
    labels_seq不是one hot，也不是字符串，而是保存的每位字符在字典中的位置。
    :param labels_seq:
    :return: label sparse
    """
    indices = []
    values = []
    for n, seq in enumerate(labels_seq):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(labels_seq), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape

def label_sequence_to_string(label_seq):
    """
    对于单个sample，label转换为字符串(true label)。
    :param label_seq: 相应位置存的是字符在字典中的位置
    :return: true label
    """
    # str = ""
    # for n, ind in enumerate(label_seq):
    #     if ind >= 0 and ind < NUM_CLASSES:
    #         str += CHARS_DICT[ind]
    return str(label_seq)

def show_one_sample_image(image,label_str):
    """
    在matplotlib显示单个图片与label
    :param image: [h,w,c]
    :param label_str: true label(string)
    :return:
    """
    # 显示图片
    # image = image.reshape(IMAGE_SHAPE)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, label_str, ha='center', va='center',transform=ax.transAxes)
    plt.imshow(image)
    plt.show()

def __decode_TFrecord(serialized_record):
    """
    从文件队列读取序列化数据TFrecord后，需要解码
    :param serialized_record: 读取的序列化数据
    :return: 每次读取单个样本 image[h,w,c], label_seq[chars_max_num], label_len, seq_len=chars_max_num
    """
    with tf.name_scope("Decode_TFRecord"):
        features = tf.parse_single_example(serialized_record,
                                           features={
                                               "image_raw": tf.FixedLenFeature([], tf.string),
                                               "image_shape_raw": tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
                                               "chars_num_raw": tf.FixedLenFeature([1], default_value=tf.zeros([1],dtype=tf.int64),dtype=tf.int64),
                                               "label_raw": tf.FixedLenFeature([CHARS_MAX_NUM],dtype=tf.int64),
                                           }, name="inputData_getSerialized")

        ## 如果是csv文件就 tf.decode_csv()
        image_raw = tf.decode_raw(features["image_raw"], tf.uint8)
        image_shape = features["image_shape_raw"]
        chars_num = tf.cast(features["chars_num_raw"], tf.int32)
        label_raw = features["label_raw"]

        seq_len = IMAGE_WIDTH
        # Image 预处理
        img = tf.reshape(image_raw, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
        img = tf.image.rgb_to_grayscale(img)        ## 变为 灰度图
        image = tf.cast(img,tf.float32) * (1 / 255) - 0.5       ## 归一化

        # label_seq = tf.cast(tf.reshape(label_raw, [CHARS_MAX_NUM]), tf.int64)
        label_seq = label_raw
        return image, label_seq, chars_num, seq_len

def get_data_from_TFrecord(is_training):
    """
    获取batch_size大小数据，
    :param is_training: 决定获取数据是训练集的还是测试集的
    :return: 给tensorflow graph提供数据
    """
    print("get_data_from_TFrecord---isTraining=", is_training, " ,tfreocrd dir=",TF_RECORD_FILE_DIR)
    if is_training:
        epochs = TRAIN_EPOCHS
        batch_size = TRAIN_BATCH_SIZE
        tfrecord_filenames_list = _get_tfrecord_filenames(TF_RECORD_FILE_DIR,key="Train")
    else:
        epochs = TEST_EPOCHS
        batch_size = TEST_BATCH_SIZE
        tfrecord_filenames_list = _get_tfrecord_filenames(TF_RECORD_FILE_DIR,key="Test")
    with tf.name_scope("Input_Data"):
        print("get_data_from_TFrecord---epochs=",epochs," ,batch_size=",batch_size," ,tfrecord_file_path=", tfrecord_filenames_list)
        # reader = tf.TextLineReader()      # 如果数据集是csv文件用这个
        if tfrecord_filenames_list == []:
            print("TFrecord文件不存在，dir:",TF_RECORD_FILE_DIR)
            exit()
        reader = tf.TFRecordReader()  # 如果数据集是TFRecord用这个
        file_queue = tf.train.string_input_producer(tfrecord_filenames_list, num_epochs=epochs, shuffle=False,
                                                    name="InputData_file_queue")
        # 从TFRecord读取序列化数据。
        _, serialized_record = reader.read(file_queue)
        # 解码序列化的数据
        image, label_seq, chars_num, seq_len = __decode_TFrecord(serialized_record)
        # Shuffle是打乱顺序。一个batch一个batch拿数据
        ### 设置了线程数量
        if is_training:
            ## batch_size是每次训练拿batch_size个数据。
            images, labels_seq, chars_num, seq_lens = tf.train.batch([image, label_seq, chars_num, seq_len],
                                                                         batch_size=batch_size,
                                                                         capacity=2000 + 3 * batch_size,
                                                                         num_threads=8, name="InputData_getBatch")
        else:
            images, labels_seq, chars_num, seq_lens = tf.train.shuffle_batch(
                [image, label_seq, chars_num, seq_len],
                batch_size=batch_size,
                capacity=2000 + 3 * batch_size,
                min_after_dequeue=2000,
                num_threads=8)

        return images, labels_seq, chars_num, seq_lens

def window_size_calculator(img_size, output_box_size):
    """
    供mdlstm计算window大小
    :param img_size: mdlstm输入数据的尺寸[?,h,w,c]
    :param output_box_size: 期望mdlstm输出的数据尺寸[?, h/win_h, w/win_w, c]
    :return: window,[win_h,win_w]
    """
    div_list = [int(x / y) for x, y in zip(img_size, output_box_size)]
    mod_list = [x % y for x, y in zip(img_size, output_box_size)]
    for i in range(len(mod_list)):
        if mod_list[i] != 0:
            div_list[i] += 1
    return div_list
