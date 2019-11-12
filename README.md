Image Recognizer
==
基于tensorflow的图片识别项目。Image->Project->Sequence。
--

## 文件作用：

* main_train.py: 训练模型的主文件，训练模型时请运行这个文件。此文件只包含Train部分，网络模型部分(反向传播)在inference.py，InputData(读取TFrecord)部分在data_tool.py
* data_tool.py: 供main_train.py调用来获取并解析TFrecord文件作为数据集。
* inference.py: 是模型文件(反向传播)，供main_train.py调用。
* image_to_tfrecord.py: 工具文件，用于获取Image数据文件并生成TFrecord。不参与模型训练。

****************
__________


`sdfsdf`

* dsfsdf


```Python
def _get_tfrecord_filenames(tfrecord_dir,key="Test"):
    label_filename_list = []
    for filename in os.listdir(tfrecord_dir):
        if filename.startswith(key) > 0:
            label_file_name = join(tfrecord_dir, filename)
            label_filename_list.append(label_file_name)
    return label_filename_list
```


    dsfsdf

#tensorflow:1.14 python:3.7.4 InputData: TFrecord Inference: CNN+MDLSTM+FC Train: CTC_loss+Adam optimizer Predict: BeamSearch  Image->Project->sequence, for example: captcha Image, Handwriting image
