"""
需要HWDB1.1trn_gnt、HWDB1.1tst_gnt、THUCNnews。THUCNnews用来生成Text语料文本，HWDB从单字Img到行img。
建立同级目录./data/test    ./data/test_info    ./data/test_text_img  ./data/text_samples 
./data/train   ./data/train_info    ./data/train_text_img

dictionary.py跟HWDB是一对一的。
"""

# 此目录下放HWDB1.1trn_gnt、HWDB1.1tst_gnt
DATA_DIR = './'
# 语料库路径
TEXT_DATA_DIR = './THUCNews'

## 图片中字数，可以固定，也可以随机
# RANDOM_CHAR_MAX_NUM = RDM.randint(5, 16)
RANDOM_CHAR_MAX_NUM = 1

## 设置生成的图片高度，宽度是HWDB自己生成的。
IMG_HEIGHT = 32
## Trai set  Test set
TRAIN_NUM = 200
TEST_NUM = 200
## 字写法
WRITE_TYPES = 4
