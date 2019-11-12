Image Recognizer
==
基于tensorflow的图片识别项目。Image->Project->Sequence。
--

********
#tensorflow:1.14 python:3.7.4 InputData: TFrecord Inference: CNN+MDLSTM+FC Train: CTC_loss+Adam optimizer Predict: BeamSearch  Image->Project->sequence, for example: captcha Image, Handwriting image
********

# 文件作用：

* **main_train.py**: 训练模型的主文件，训练模型时请运行这个文件。此文件只包含Train部分，网络模型部分(反向传播)在inference.py，InputData(读取TFrecord)部分在data_tool.py
* **data_tool.py**·: 供main_train.py调用来获取并解析TFrecord文件作为数据集。
* **inference.py**: 是模型文件(反向传播)，供main_train.py调用。
* **image_to_tfrecord.py**: 工具文件，用于获取Image数据文件并生成TFrecord。不参与模型训练。
***
# 使用：
## 1. 准备数据生成TFrecord数据集(通过image_to_tfrecord.py工具把Image批量转换为TFrecord数据集)
### 1.1. 先读取label_info_txt.
　　**使用时，只需修改LABEL_INFO_PATH为label_info.txt所在的目录地址，支持多label_info.txt同时读取。**但注意label中的image地址正确。TF_RECORD_FILE_NAME修改为想要保存的TFrecord文件路径(是期望生成的TFrecord文件所在目录)，TF_RECORD_FILE_NAME是期望生成的TFrecord文件名，**注意"Train"是训练集的关键字，想做为训练集时TFrecord文件名要带"Train"，Test同理。**然后运行代码即可.

```Python
 # image_to_tfrecord.py
LABEL_INFO_PATH = "/Users/hly/PycharmProjects/HWR_1112/data/test_info" ## label info.txt所在的目录，支持多info.txt同时读取
TF_RECORD_FILE_NAME = "TrainTFrecord"      ## 生成的tfrecord文件名,注意'Train'是关键字，data_tool.py获取到文件名列表中带有'Train'关键字就会认为此文件是Train set。'Test'关键字作为测试集。

TF_RECORD_FILE_DIR = "./data/tfrecord/"  ## 生成的TFrecord地址
```

　　label_info.txt内容格式如下，每一行都是一个样本，第一位是Image路径，第二位是label，注意Image相对地址是否正确：

```Python
 # label_info.txt
['./gen_images/187.png', [1756, 1471, 1060, 1023, 3266, 88, 98]]
['./gen_images/188.png', [2267, 2271, 155, 1518, 2392, 2268, 1474]]
['./gen_images/189.png', [1524, 3299, 1045, 2813, 3649, 1234, 212]]
```
<img src="https://github.com/holny/Image_Recognizer/raw/master/Other/Image/WechatIMG7.png" width="500">

　　**LABEL_INFO_PATH是指定label_info所在的目录，不是label_info本身。根据目录可以同时读取多个存在于此目录下的label_info并且生成TFrecord。**
  
 
 <img src="https://github.com/holny/Image_Recognizer/blob/master/Other/Image/WX20191112-201704%402x.png" width="500" height="200"> 
 
 ### 1.2. 根据label_info.txt里的Image地址信息读取Image与label，并生成TFrecord
 
 　　**因为Image与Label存入TFrecord统一尺寸，所以Image的Height高于以下时会被缩小Height，当width不足以下时会填充，padding255(白色)。 label会被统一长度为15(单个label中最大字符数)，不足的padding 0(CTC_LOSS计算，0为算作为空位)：**

```Python
 # image_to_tfrecord.py
IMAGE_HEIGHT = 32       ## 在存入tfrecord之前固定Image尺寸，不足padding
IMAGE_WIDTH = 600
CHARS_MAX_NUM = 15      ## 15, 单个label中最大字符数，在label存入tfrecord之前固定长度为15，不足padding
```
 
　　最后生成的TFrecord文件会自动在文件名后附加数据条数和生成月日。
  
 <img src="https://github.com/holny/Image_Recognizer/blob/master/Other/Image/WX20191112-201421%402x.png" width="500" height="400">
 
 
## 2. 运行main_train.py文件，读取TFrecord训练模型.
 ### 2.1. 首先注意TFrecord文件目录设置正确。
 
 　　**main_train.py调用data_tool.py来解析TFreocrd文件，获取数据**
  
    
 　　可以同时读取此目录下**多个TFrecord文件**作为文件名队列解析TFrecord。注意TF_RECORD_FILE_DIR**是TFrecord文件所在的目录。**
 ```Python
 # data_tool.py
 TF_RECORD_FILE_DIR = image_to_record.TF_RECORD_FILE_DIR    # TFrecord文件所在目录，这里就直接调用image_to_record.py中的生成路径。
 ```

　　**如何区分Train set与Test set?**
　　首先注意运行main_train.py前，可以设置这次训练模型是’训练‘还是'测试'，如下注释。生成HW_Recognizer对象传递的参数决定了 **IS_TRAINING**的值。data_tool.py会根据IS_TRAINING的值来判断这次是'训练'，还是’测试‘。
  
```Python
## main_train.py中最后一行
nn = HW_Recognizer(True)    ## 训练时填入True，测试时填入False
nn.train()
```
　　如下，当is_training=True(说明是在训练)，**则过滤TF_RECORD_FILE_DIR目录下的文件名带有关键字"Train"的所有TFrecord文件名作为List列表队列**(所以前面生成TFrecord时，TF_RECORD_FILE_NAME要有关键字)。传入string_input_producer()读取文件名队列。以此实现读取多TFrecord。 也以此实现了区分Train set与 Test set。

```Python
# data_tool.py -> get_data_from_TFrecord(is_training)
 if is_training:
        epochs = TRAIN_EPOCHS
        batch_size = TRAIN_BATCH_SIZE
        tfrecord_filenames_list = _get_tfrecord_filenames(TF_RECORD_FILE_DIR,key="Train")
 else:
        epochs = TEST_EPOCHS
        batch_size = TEST_BATCH_SIZE
        tfrecord_filenames_list = _get_tfrecord_filenames(TF_RECORD_FILE_DIR,key="Test")
        
reader = tf.TFRecordReader()                   
file_queue = tf.train.string_input_producer(tfrecord_filenames_list, num_epochs=epochs, shuffle=False,
                                            name="InputData_file_queue")
```
 
　　**因此区分Train set与Test set的不是靠目录路径，而是靠TFrecord本身的文件名**
  
### 2.2. 设置好了TF_RECORD_FILE_DIR，就运行main_train.py开始训练模型。

# 相关参数设置：

## 1. checkponit保存模型参数。

　　注意IS_NEED_SAVE在'训练'模式（IS_TRAINING=True），'测试'模式（IS_TRAINING=False）,**不同模式设置不同的IS_NEED_SAVE。因为'测试'模式，只是在已训练好的模型上用Test set看模型准确率，并不进行反向传播，也就不进行保存模型参数了**

　　**如果训练时比较卡，可以关闭保存参数。**
  　**MODEL_PATH Train模型与Test模型地址一样，因为Test时候会读取Train完成后的模型参数数据，以此为基础对Test set进行预测。**

```Python
# main_train.py
 if self.IS_TRAINING:  ## 训练时的参数
        self.IS_NEED_SAVE = True  ## 是否保存训练模型数据
        self.MODEL_PATH = "./ModelTrain"  ## 模型数据保存地址
 else:  #### 测试时参数
        self.IS_NEED_SAVE = False
        self.MODEL_PATH = "./ModelTrain"  ## 模型数据保存地址
```
## 2. Tensorboard显示。

　　IS_SUMMARY = False并不关闭Tensorboard，只是不再记录inference中的各种参数，而整个网络模型的graph还是会显示的。注意受IS_TRAINING影响。

```Python
# infernece.py
 if IS_TRAINING:  ## 训练时的参数
       IS_SUMMARY = True
 else:  #### 测试时参数
       IS_SUMMARY = False
```
```Python
# main_train.py
self.LOG_PATH = "./SummaryTrain"  ## Tensorboard Log保存地址
```
## 3. 实现了动态学习率(指数下降学习率)。

　　都在main_train.py中设置，如下图：
　　**想让学习率不变时，只需设置DECAY_RATE=1，学习率就一直是START_LEARNING_RATE**
  
<img src="https://github.com/holny/Image_Recognizer/blob/master/Other/Image/WX20191112-210327%402x.png" width="500" height="300">

## 4. epochs(训练轮数)，batch_size设置。
　　data_tool.py中设置。
　　**epochs是训练完整个TFrecord算作一轮，而batch_size是一次训练(一次正向+反向)从TFrecord抓取多少条数据**
　　
　　**Train模式与Test模式使用不同的epochs与batch_size，需要分别设置**
  
　　**另外说下，如下图(data_tool.py)可以设置NUM_CLASSES，这是字符种类数(需要+1，为CTC留0作空位)。**
  
<img src="https://github.com/holny/Image_Recognizer/blob/master/Other/Image/WX20191112-210412%402x.png" width="500" height="300">

## 5. Dropout、L2正则化设置。
　　inference.py中设置。
　　**P_KEEP_CONV与P_KEEP_FC分别设置卷积层中和最后的FC中的dropout rate(全1代表都保留，不dropout)。受IS_TRAINING影响，因为Test模式时不要进行Dropout**
　　**IS_REGULARIZER设置是否开启L2正则化，REGULARIZATION_RATE设置正则化系数。**

## 6. CNN层、MDLSTM层、FC层中设置神经单元数。
　　都在inference.py，如下图：
　　**注意，修改时要注意层与层之间的shape是否对接的上，要计算推一遍过程**

<img src="https://github.com/holny/Image_Recognizer/blob/master/Other/Image/WX20191112-210358%402x.png" width="500" height="300">


# 不足以及待改进地方-20191112：
### 1. GPU需要适配，虽然了解到现在也是在显存上跑，但tensorflow的确是可以设置的，所以如果跑不动，看是否在GPU上跑。
### 2. github上其它项目，都有设置Flag参数(具体什么忘了)，作用就是在控制台运行python文件时可以附带参数，比如指定路径什么的，也就不用再.py文件里改了。当然如果现在不影响什么可以不管，不过我后面上云时候有闲事搞下。
### 3. MDLSTM的dropout也需要实现下，MDLSTM里面也是有参数参与训练的，不过现在用不着。
