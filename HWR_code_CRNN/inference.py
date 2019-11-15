import tensorflow as tf
import data_tool as data_tool, md_lstm as md_lstm
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib import slim

NUM_CLASSES = data_tool.NUM_CLASSES  ## 为空位+1，0代表空位
## 正则化regularizer
IS_REGULARIZER = False  ## 是否开启L2正则化
REGULARIZATION_RATE = 1e-3  ## 正则化系数


### CRNN
## CNN layer    注意seq_len也要相应改变
# CONV_PADDING_SAME = "SAME"  ## 卷积层1,卷积padding,SAME不改变h,w
# POOL_PADDING = "SAME"  ## 池化padding类型
# CONV_STRIDES = (1,1)  ## 卷积层1,卷积步长，中间两位为准:1x1
CONV_ACTIVATION = tf.nn.relu

CONV_KERNEL_NUM_1 = 64  ## 卷积层1,卷积核个数
CONV_KERNEL_SIZE_1 = (3,3)  ## 卷积层1,卷积核大小，输入的是灰度图，通道1
POOL_SIZE_1 = (2,2)  ## 池化核大小，2x2
POOL_STRIDES_1 = (2,2)  ## 池化步长.2x2

CONV_KERNEL_NUM_2 = 128
CONV_KERNEL_SIZE_2 = (3,3)
POOL_SIZE_2 = (2,2)
POOL_STRIDES_2 = (2,2)

CONV_KERNEL_NUM_3 = 256
CONV_KERNEL_SIZE_3 = (3,3)
POOL_SIZE_3 = (2,1)
POOL_STRIDES_3 = (2,1)

CONV_KERNEL_NUM_4 = 256
CONV_KERNEL_SIZE_4 = (3,3)

CONV_KERNEL_NUM_5 = 512
CONV_KERNEL_SIZE_5 = (3,3)
POOL_SIZE_5 = (2,1)
POOL_STRIDES_5 = (2,1)

CONV_KERNEL_NUM_6 = 512
CONV_KERNEL_SIZE_6 = (2,2)
## BiLSTM
BILSTM_HIDDEN_NUM_UNITS_1 = 256
BILSTM_HIDDEN_NUM_UNITS_2 = 256

def _get_config(IS_TRAINING):
    if IS_TRAINING:  ## 训练时的参数
        config = {
            ## CNN
            "P_KEEP_CONV":1,    ## 卷积层1中Dropout
            "P_KEEP_BILSTM":1,
            ## FC
            "P_KEEP_FC":1,
            "IS_SUMMARY":False, ## 是否进行在Tensorboard上显示网络中数据，graphs还是会显示的。
            "P_INPUT_KEEP_PROB":0.9,
            "P_OUTPUT_KEEP_PROB":0.9
        }
    else:
        config = {
            ## CNN
            "P_KEEP_CONV": 1,
            "P_KEEP_BILSTM": 1,
            ## FC
            "P_KEEP_FC": 1,
            "IS_SUMMARY": False,
            "P_INPUT_KEEP_PROB":1,
            "P_OUTPUT_KEEP_PROB":1
        }

    return config

def get_weight( name, shape ,stddev = 0.1, mean= 0):
    initializer = tf.truncated_normal_initializer(stddev=stddev, mean=mean)
    if IS_REGULARIZER:
        # L2正则化--1/2--对权重w添加l2正则化，tensorflow会将变量加入集合 tf.GraphKeys.REGULARIZATOIN_LOSSES
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        return tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=tf.float32,
                               regularizer=regularizer)
    else:
        return tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=tf.float32)

def get_bias(name, shape):
    initializer = tf.zeros_initializer()
    return tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=tf.float32)

def cnn_net(inputs,seq_lens,is_training):  # CRNN-CNN
    config = _get_config(is_training)
    ## seq_lens 要始终等于time_Step,而卷积池化会改变time_Step所以seq_lens也要改变[?,time_step]
    shape = inputs.get_shape().as_list()
    input_h, input_w = shape[1], shape[2]
    print("cnn_net--inputs.shape=", shape," ,in training:",is_training)
    with tf.variable_scope("Conv_MaxPool_layer1"):
        conv1_p = tf.layers.conv2d(inputs=inputs, filters=CONV_KERNEL_NUM_1, kernel_size=CONV_KERNEL_SIZE_1,
                                 padding="SAME", activation=CONV_ACTIVATION, name="conv_1")
        conv1 = tf.nn.dropout(conv1_p, config["P_KEEP_CONV"])
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=POOL_SIZE_1, strides=POOL_STRIDES_1, name="pool_1")
        # seq_lens = tf.div(seq_lens,2)            ## 因为 池化改变了time_Step
        print("cnn_net--layer1--out.shape=",pool1.get_shape().as_list())    # (?,16,w/2,64)  高固定32
    with tf.variable_scope("Conv_MaxPool_layer2"):
        conv2_p = tf.layers.conv2d(inputs=pool1, filters=CONV_KERNEL_NUM_2, kernel_size=CONV_KERNEL_SIZE_2,
                                   padding="SAME", activation=CONV_ACTIVATION, name="conv_2")
        conv2 = tf.nn.dropout(conv2_p, config["P_KEEP_CONV"])
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=POOL_SIZE_2, strides=POOL_STRIDES_2, name="pool_2")
        # seq_lens = tf.div(seq_lens, 2)  ## 因为 池化改变了time_Step
        print("cnn_net--layer2--out.shape=", pool2.get_shape().as_list())   # (?,8,w/2,128)
    with tf.variable_scope("Conv_MaxPool_layer3"):
        conv3_p = tf.layers.conv2d(inputs=pool2, filters=CONV_KERNEL_NUM_3, kernel_size=CONV_KERNEL_SIZE_3,
                                   padding="SAME", activation=CONV_ACTIVATION, name="conv_3")
        conv3 = tf.nn.dropout(conv3_p, config["P_KEEP_CONV"])
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=POOL_SIZE_3, strides=POOL_STRIDES_3, name="pool_3")
        # seq_lens = tf.div(seq_lens, 2)  ## 因为 池化改变了time_Step
        print("cnn_net--layer3--out.shape=", pool3.get_shape().as_list())   # (?,4,w,256)
    with tf.variable_scope("Conv_BN_layer4"):
        conv4_p = tf.layers.conv2d(inputs=pool3, filters=CONV_KERNEL_NUM_4, kernel_size=CONV_KERNEL_SIZE_4,
                                   padding="SAME", activation=CONV_ACTIVATION, name="conv_4")
        conv4 = tf.nn.dropout(conv4_p, config["P_KEEP_CONV"])
        bn1 = tf.layers.batch_normalization(conv4,training=is_training,name="batch_normal1")
        # seq_lens = tf.div(seq_lens, 2)  ## 因为 池化改变了time_Step
        print("cnn_net--layer4--out.shape=", bn1.get_shape().as_list())     # (?,4,w,256)
    with tf.variable_scope("Conv_BN_MaxPool_layer5"):
        conv5_p = tf.layers.conv2d(inputs=bn1, filters=CONV_KERNEL_NUM_5, kernel_size=CONV_KERNEL_SIZE_5,
                                   padding="SAME", activation=CONV_ACTIVATION, name="conv_5")
        conv5 = tf.nn.dropout(conv5_p, config["P_KEEP_CONV"])
        bn2 = tf.layers.batch_normalization(conv5, training=is_training, name="batch_normal2")
        pool5 = tf.layers.max_pooling2d(inputs=bn2, pool_size=POOL_SIZE_5, strides=POOL_STRIDES_5, name="pool_5")
        # seq_lens = tf.div(seq_lens, 2)  ## 因为 池化改变了time_Step
        print("cnn_net--layer5--out.shape=", pool5.get_shape().as_list())   # (?,2,w,512)
    with tf.variable_scope("Conv_layer6"):
        conv6_p = tf.layers.conv2d(inputs=pool5, filters=CONV_KERNEL_NUM_6, kernel_size=CONV_KERNEL_SIZE_6,
                                   padding="VALID", activation=CONV_ACTIVATION, name="conv_6")
        conv6 = tf.nn.dropout(conv6_p, config["P_KEEP_CONV"])
        # seq_lens = tf.div(seq_lens, 2)  ## 因为 池化改变了time_Step
        print("cnn_net--layer6--out.shape=", conv6.get_shape().as_list())
    seq_lens = tf.constant(shape=[shape[0],],value=conv6.get_shape().as_list()[2])      ## 固定seq_lens.shape=(?,) ,value = width
    print("cnn_net--final-seq_lens.shape=",seq_lens.shape)

    with tf.variable_scope("Map_to_Seq"):
        cnn_output = tf.squeeze(conv6, axis=1)      ## 因为输入height固定32，卷积池化导致(?,1,100->24,512)
        print("cnn_net--map to seq--out.shape=", cnn_output.get_shape().as_list())

    if config["IS_SUMMARY"]:
        # Tensorboard-用于可视化Conv-MaxPool后的图像
        tf.summary.image("cnn_inputs", inputs, 5)  # 用于Tensorboard可视化显示
        # conv1_for_show1 = tf.reshape(conv1_a[:, :, :, 0], [-1, input_h, input_w, 1])
        # conv1_for_show2 = tf.reshape(conv1_a[:, :, :, 1], [-1, input_h, input_w, 1])
        # conv1_for_show3 = tf.reshape(conv1_a[:, :, :, 2], [-1, input_h, input_w, 1])
        # tf.summary.image("conv1_for_show1", conv1_for_show1, 5)  # 输出带有5张图像的summary协议缓冲区，channel1
        # tf.summary.image("conv1_for_show2", conv1_for_show2, 5)  # 输出带有5张图像的summary协议缓冲区，channel2
        # tf.summary.image("conv1_for_show3", conv1_for_show3, 5)  # 输出带有5张图像的summary协议缓冲区，channel3
    return cnn_output,seq_lens


def bi_lstmm_net(inputs,seq_lens,is_training):       ##  CRNN - BiLSTM
    config = _get_config(is_training)
    shape = inputs.get_shape().as_list()    # [?, width, features]    height经过卷积池化再squeeze()后降维了
    batch_size = shape[0]
    print("bi_lstmm_net--inputs.shape=", shape," ,is_training=",is_training)
    with tf.variable_scope('BiLSTM_layer1'):
        lstm_fw_cell_1 = BasicLSTMCell(BILSTM_HIDDEN_NUM_UNITS_1)
        lstm_fw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell_1, input_keep_prob=config["P_INPUT_KEEP_PROB"], output_keep_prob=config["P_OUTPUT_KEEP_PROB"])
        lstm_bw_cell_1 = BasicLSTMCell(BILSTM_HIDDEN_NUM_UNITS_1)
        lstm_bw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell_1, input_keep_prob=config["P_INPUT_KEEP_PROB"], output_keep_prob=config["P_OUTPUT_KEEP_PROB"])
        bilstm_output1, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1,
                                                          lstm_bw_cell_1,
                                                          inputs, seq_lens,
                                                          dtype=tf.float32)
        # print("bi_lstmm_net--bilstm_output1.shape=",bilstm_output1)
        ## bidirectional_dynamic_rnn输出的 bilstm_output1 是前向后项两个，分别shape=(?,w,hidden_num)。需要连接起来
        lstm_layer_output1 = tf.concat(bilstm_output1, 2)     # shape=(?,w,hidden_num*2)
        # print("bi_lstmm_net--lstm_layer_output1.shape=", lstm_layer_output1.get_shape().as_list())

    with tf.variable_scope('BiLSTM_layer2'):
        lstm_fw_cell_2 = BasicLSTMCell(BILSTM_HIDDEN_NUM_UNITS_2)
        lstm_fw_cell_2 = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell_2, input_keep_prob=config["P_INPUT_KEEP_PROB"], output_keep_prob=config["P_OUTPUT_KEEP_PROB"])
        lstm_bw_cell_2 = BasicLSTMCell(BILSTM_HIDDEN_NUM_UNITS_2)
        lstm_bw_cell_2 = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell_2, input_keep_prob=config["P_INPUT_KEEP_PROB"], output_keep_prob=config["P_OUTPUT_KEEP_PROB"])
        bilstm_output2, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2,
                                                          lstm_bw_cell_2,
                                                          inputs, seq_lens,
                                                          dtype=tf.float32)
        # print("bi_lstmm_net--bilstm_output2.shape=", bilstm_output2)
        ## bidirectional_dynamic_rnn输出的 bilstm_output1 是前向后项两个，分别的shape=(?,w,hidden_num)。需要连接起来
        lstm_layer_output2 = tf.concat(bilstm_output2, 2)     # shape=(?,w,hidden_num*2)
        print("bi_lstmm_net--lstm_layer_output2.shape=", lstm_layer_output2.get_shape().as_list())

    with tf.variable_scope('Transcription_layer'):
        rnn_reshaped = tf.reshape(lstm_layer_output2, shape=[-1, BILSTM_HIDDEN_NUM_UNITS_2*2])
        # doing the affine projection
        trans_w = get_weight(name="trans_w",shape=[512, NUM_CLASSES])
        trans_b = get_bias(name="trans_b",shape=[NUM_CLASSES])
        logits = tf.matmul(rnn_reshaped, trans_w) + trans_b
        logits = tf.reshape(logits, shape=[batch_size, -1, NUM_CLASSES])
        # final layer, the output of BLSTM
        rnn_net_output = tf.transpose(logits, (1, 0, 2), name='transpose_time_major')
        print("bi_lstmm_net--rnn_net_output.shape=", rnn_net_output.get_shape().as_list())

    if config["IS_SUMMARY"]:
        # tf.summary.image("cnn_input_image", inputs, 5)  # 用于Tensorboard可视化显示，
        tf.summary.histogram("Trans/weights", trans_w)
        tf.summary.histogram("Trans/biases", trans_b)
        # Tensorboard-用于可视化Conv-MaxPool后的图像
        tf.summary.image("rnn_inputs", inputs, 5)  # 用于Tensorboard可视化显示
    return rnn_net_output
