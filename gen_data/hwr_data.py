# -*- coding:utf-8 -*-
#@Time : 2019-11-11 22:56
#@Author: xbb1973
#@File : hwr_data.py
import chinese_character_recognition.dictionary
import os
from os import walk
import numpy as np
import struct
from PIL import Image
import pickle
import re
import time

# 测试专用，测试无误后可打开生成全部数据
is_test = False
# Pycharm用户记得在Preference中的project structure将数据文件exclude，否则会卡死！！！！

RDM = np.random.RandomState(seed=2)


class DataGenerator:
    def __init__(self):
        self.char_dict = dict(chinese_character_recognition.dictionary.char_dict)

    def gen_single_character_from_HWDB(self):
        '''
        获取单张图片数据集
        HWDB训练数据集路径train_data_dir
        HWDB测试数据集路径test_data_dir
        :return:
        '''
        train_counter = 0
        test_counter = 0
        data_dir = '../../res'
        train_data_dir = os.path.join(data_dir, 'HWDB1.1trn_gnt')
        test_data_dir = os.path.join(data_dir, 'HWDB1.1tst_gnt')
        train_save_path = './data/train/'
        test_save_path = './data/test/'
        for image, tagcode in self.read_from_gnt_dir(gnt_dir=train_data_dir):
            tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
            im = Image.fromarray(image)
            dir_name = train_save_path + '%d' % self.char_dict[tagcode_unicode]
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            im.convert('RGB').save(dir_name + '/' + str(train_counter) + '.png')
            print("train_counter=", train_counter)
            train_counter += 1
            if is_test:
                if train_counter > 10240:
                    break
            # 0-897757

        for image, tagcode in self.read_from_gnt_dir(gnt_dir=test_data_dir):
            tagcode_unicode = struct.pack('>H', tagcode).decode('gb2312')
            im = Image.fromarray(image)
            dir_name = test_save_path + '%d' % self.char_dict[tagcode_unicode]
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            im.convert('RGB').save(dir_name + '/' + str(test_counter) + '.png')
            print("test_counter=", test_counter)
            test_counter += 1
            if is_test:
                if test_counter > 10240:
                    break
            # 0-223990

    def read_from_gnt_dir(self, gnt_dir):
        '''
        获取HWDB1.1trn_gnt和HWDB1.1tst_gnt的图片数据
        :param gnt_dir:gnt路径
        :return:image, tagcode
        '''
        def one_file(f):
            header_size = 10
            while True:
                header = np.fromfile(f, dtype='uint8', count=header_size)
                if not header.size: break
                sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
                tagcode = header[5] + (header[4] << 8)
                width = header[6] + (header[7] << 8)
                height = header[8] + (header[9] << 8)
                if header_size + width * height != sample_size:
                    break
                image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
                yield image, tagcode

        for file_name in os.listdir(gnt_dir):
            if file_name.endswith('.gnt'):
                file_path = os.path.join(gnt_dir, file_name)
                with open(file_path, 'rb') as f:
                    for image, tagcode in one_file(f):
                        yield image, tagcode

    def get_root_dir_file(self, file_dir):
        '''
        遍历文件路径
        :param file_dir:
        :return:
        '''
        for root, dirs, files in walk(file_dir):
            return root, dirs, files

    def handle_text(self):
        '''
        处理文本语料库，获得具有意义的文本；
        正则表达式匹配中文字符，去除标点符号等非中文字符。
        文本数据路径text_data_dir
        处理后的文本样本路径text_samples_root_dir
        :return:
        '''
        text_data_dir = '../../res/THUCNews'
        text_samples_root_dir = './data/text_samples'

        text_data_root, text_data_dirs, text_data_files = self.get_root_dir_file(text_data_dir)
        # print(text_data_root)
        # print(text_data_dirs)
        # print(text_data_files)
        # ../../res/THUCNews
        # ['时尚', '家居', '教育', '股票', '娱乐', '彩票', '社会', '房产', '星座', '科技', '财经', '时政', '游戏', '体育']
        # []
        count = 0
        for dir in text_data_dirs:
            dir_path = text_data_root + '/' + dir
            # print(dir_path)
            text_data_dirs, _, text_data_files = self.get_root_dir_file(dir_path)
            # print(text_data_root)
            # print(text_data_dirs)
            # print(text_data_files)
            # ../../res/THUCNews/时尚
            # []
            # ['333620.txt', '335251.txt', '329678.txt', '331037.txt', '336758.txt', '338775.txt'

            if True or is_test:
                if count > 50:
                    break
            for file in text_data_files:
                file_path = text_data_dirs + '/' + file
                print(file_path)
                with open(file_path, 'r', encoding='utf-8') as f1:
                    # 读取所有，然后再按行分割，这样读取的list不会带有换行符
                    text = f1.read()
                    # 去除空格和\xa0、\u3000
                    text.strip().replace(u'\u3000', u' ').replace(u'\xa0', u' ').replace(u'\r', u' ')
                # print(text)

                # 正则匹配中文，固定形式：\u4E00 -\u9FA5
                pattern = re.compile(r"[\u4e00-\u9fa5]+")
                result = pattern.findall(text)
                # print(result)
                new_text = ''.join(result)
                print(new_text)

                text_samples_path = text_samples_root_dir + '/' + str(count) + '.txt'
                f1 = open(text_samples_path, 'w')
                f1.write(new_text)

                print(count)
                count += 1
                # 836074
                pass

    def get_random_text(self):
        '''
        随机获取文本数据
        :return:text
        '''
        text_samples_root, text_samples_dirs, text_samples_files = self.get_root_dir_file('./data/text_samples')
        # print(text_samples_root)
        # print(text_samples_dirs)
        # print(text_samples_files)
        # ./text_samples
        # []
        # ['1.txt']
        text_file_path = text_samples_root + '/' + RDM.choice(text_samples_files)
        # print(text_file_path)
        # ./text_samples/1.txt
        with open(text_file_path, 'r', encoding='utf-8') as f1:
            # 目前只读一行，因为处理后读文本只有一行
            text = f1.readline()
        # print(text)
        return text

    def get_char_img(self, char):
        '''
        生成单字图片
        :param char:
        :return: char_img
        '''
        char_dir = str(self.char_dict[char])
        char_dir, _, char_file = self.get_root_dir_file(self.char_img_root + '/' + char_dir)
        # print(_)
        # print(char_dir)
        # print(char_file)
        # []
        # ./train/1579
        # ['3323.png', '9543.png', '4552.png']
        # ./train/1579/9543.png
        char_path = char_dir + '/' + RDM.choice(char_file)
        # print(char_path)
        while True:
            try:
                char_img = Image.open(char_path)
                break
            except:
                char_path = char_dir + '/' + RDM.choice(char_file)

        # 控制字体高度
        char_size = int(self.height-4)
        if char_img.height > char_img.width:
            size_rate = char_size / char_img.height
        else:
            size_rate = char_size / char_img.width
        char_img = char_img.resize(
            (int(char_img.width * size_rate), int(char_img.height * size_rate)))
        # char_img = char_img.resize(char_size, char_size)

        # img = cv2.copyMakeBorder(img, 0, 0, shape_offset, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        background = Image.new('RGB', (int(char_img.width), char_img.height),
                         (255, 255, 255))
        background.paste(char_img)
        return background

    def get_str_img(self, str):
        '''
        生成文本行图片
        :param str:
        :return: str_img
        '''
        # modify by hongyilin, change height to self.height*2
        # width=字符串长度*2*self.height*2
        # 当前self.height=48
        str_img_width = int(len(str) * (2)) * self.height * 2
        str_img_height = self.height
        str_img = Image.new('RGB',
                         (str_img_width, str_img_height),
                         (255, 255, 255))
        char_width = self.height

        for char in str:
            char_img = self.get_char_img(char)
            # 拼接汉字字符，这里可以控制间距进行数据增强
            # augment place
            str_img.paste(char_img, (char_width,  int((str_img.height - char_img.height) / 2)))
            char_width += char_img.width
        str_img = str_img.crop((0, 0, char_width+self.height, str_img_height))
        return str_img

    def get_char_list(self, str):
        '''
        分解str，获得单个char，再根据char_dict得到每一个char的对应编号，最终等到char_list
        :param str:
        :return:char_list:str中每个char的编号列表
        '''
        char_list = []
        new_str = ''
        for char in str:
            try:
                char_list.append(self.char_dict[char])
                new_str += char
            except:
                pass
        return new_str, char_list

    def gen_data_with_img_and_label(self, str, img_height, char_img_root, augment=False):
        '''
        获取最终数据
        :param str:文本行
        :param img_name:图片保存名称
        :param data_path:图片保存地址
        :param augment:是否启用数据增强
        :param is_test:是否为测试状态
        :return:img, label
        '''

        self.height = img_height
        self.augment = augment
        self.char_img_root = char_img_root
        self.char_dict = chinese_character_recognition.dictionary.char_dict

        str, char_list = self.get_char_list(str)
        img = self.get_str_img(str)

        if self.augment:
            line = len(str) + 2
            width_num = line
            for x in range(int(line)):
                startX = img.size[0] / int(line) * (x + 1)
                startY = 0
                endX = startX
                endY = img.size[1]
                draw.line([(startX, startY), (endX, endY)], fill="gray", width=1)
            line = 6
            height_num = line
            for x in range(int(line)):
                startX = 0
                startY = img.size[1] / int(line) * (x + 1)
                endX = img.size[0]
                endY = startY
                draw.line([(startX, startY), (endX, endY)], fill="gray", width=1)
            img = self.iaa_augment(img, width_num, height_num)

        return img, char_list

    # add by hly for augment
    def iaa_augment(self, image, width_num, height_num):
        # 整体流程为：定义变换序列（Sequential）→读入图片（imread）→执行变换（augment_images）→保存图片（imwrite）
        # imgaug test
        # StochasticParameter
        aug_pwa = iaa.PiecewiseAffine(scale=(0.01, 0.011), nb_rows=(2, height_num), nb_cols=(2, width_num),
                                      order=1, cval=0, mode='constant',
                                      absolute_scale=False, polygon_recoverer=None, name=None,
                                      deterministic=False, random_state=None)

        # 在每个图像上放置规则的点网格，然后将每个点随机地移动2 - 3％
        # aug = iaa.PiecewiseAffine(scale=(0.02, 0.03))

        seq = iaa.Sequential([
            # iaa.Crop(px=(0, 16)),  # 从每侧裁剪图像0到16px（随机选择）
            # iaa.Fliplr(0.5),  # 水平翻转图像
            # iaa.GaussianBlur(sigma=(0, 3.0))  # 使用0到3.0的sigma模糊图像
            aug_pwa
        ])

        imglist = []
        # img = cv2.imread('./gen_images/0.png')
        # img为opencv，image为PIL，二者进行转化
        # PIL->opencv
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        imglist.append(img)
        images_aug = seq.augment_images(imglist)
        # aug.draw_grid(img, rows=4, cols=4)
        # aug_pwa.show_grid(img, rows=2, cols=2)
        # cv2.imwrite('./gen_images/0_augement.png', images_aug[0])

        # opencv->PIL
        image = Image.fromarray(cv2.cvtColor(images_aug[0], cv2.COLOR_BGR2RGB))
        return image


    def normalizer_image(self, img_path):
        img = cv2.imread(img_path, 0)
        ar = np.array(img)
        max_width = 568
        shape_offset = (max_width - ar.shape[1])
        # BLACK = [0,0,0]
        img = cv2.copyMakeBorder(img, 0, 0, shape_offset, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        # 根据需求再改
        return img

    def gen_img_and_info(self, img_total_size, write_types, save_path, img_height, char_img_root, info_path):
        gen_img_count = 0
        labels = []
        # 每次循环输出8种不同写法，暂时不加入数据增强部分，考虑后续重新生成带数据增强的数据集作为对比方案。
        while gen_img_count < img_total_size:
            # 每次从随机文本中从random_text_start_pos开始取random_char_max_num个汉字
            text = self.get_random_text()
            text_len = len(text)
            random_char_max_num = RDM.randint(5, 16)
            # 设置定长为7
            # random_char_max_num = 7
            random_text_start_pos = RDM.randint(0, text_len)
            str_gen = text[random_text_start_pos:random_text_start_pos+random_char_max_num]
            # print(str_gen)

            key_error_count = 0
            for char_item in str_gen:
                try:
                    char_file_dir = self.char_dict[char_item]
                except KeyError:
                    key_error_count += 1
                    print(key_error_count)
                    char_file_dir = RDM.randint(0, 3755)

                # print(char_file_dir)
                # break
                # str_gen += char_dict_item.split(' ')[1]
                # print(str_gen)

            # 同一个文本行输出write_types种不同写法，暂时不加入数据增强部分。
            for i in range(write_types):
                img, char_list = dg.gen_data_with_img_and_label(str=str_gen,
                                            img_height=img_height,
                                            char_img_root=char_img_root,
                                            augment=False)

                save_addr = save_path + str(gen_img_count) +'.png'
                print(save_addr)
                img.save(save_addr)
                # modify by hongyilin for gen data
                # line = [addr, label, box]
                line = [save_addr, char_list]
                labels.append(line)
                print(gen_img_count)
                gen_img_count += 1
            # 最后将标签写入文件，考虑到数据太多，如果最后再写可能会出问题，现在每一次循环追加的形式写一次
            # 32位的py list可以存储5000w个元素，64位的py list可以存储......
            with open(info_path, 'w', encoding='UTF-8') as f1:
                for i in range(0, len(labels)):
                    line1 = labels[i]
                    # print(line1)
                    # print(str(line1))
                    f1.write(str(line1))
                    f1.write('\n')

if __name__ == '__main__':
    dg = DataGenerator()

    # HWDB数据解析，到对应api下修改相关路径，相关路径暂时hardcode，不做参数使用
    # data_dir = '../../res'
    # train_data_dir = os.path.join(data_dir, 'HWDB1.1trn_gnt')
    # test_data_dir = os.path.join(data_dir, 'HWDB1.1tst_gnt')
    # train_save_path = './data/train/'
    # test_save_path = './data/test/'
    # dg.gen_single_character_from_HWDB()

    # text数据解析，到对应api下修改相关路径，相关路径暂时hardcode，不做参数使用
    # text_data_dir = '../../res/THUCNews'
    # text_samples_root_dir = './data/text_samples'
    # dg.handle_text()

    img_height = 32
    # 修改生成数据集总大小
    img_total_size = 2048
    write_types = 4
    train_text_imge_path = './data/train_text_img/'
    train_char_img_root = './data/train'
    train_info_path = './data/train_info/info.txt'
    dg.gen_img_and_info(img_total_size, write_types, train_text_imge_path,
                        img_height, train_char_img_root, train_info_path)



    test_text_imge_path = './data/test_text_img/'
    test_char_img_root = './data/test'
    test_info_path = './data/test_info/info.txt'
    dg.gen_img_and_info(img_total_size, write_types, test_text_imge_path,
                        img_height, test_char_img_root, test_info_path)
