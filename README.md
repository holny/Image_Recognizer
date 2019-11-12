Image Recognizer
==
基于tensorflow的图片识别项目。Image->Project->Sequence。
--
    
## 文件作用：
* main_train.py
=

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
