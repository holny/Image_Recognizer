import tensorflow as tf
import data_tool as data_tool, md_lstm as md_lstm
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib import slim

IS_TRAINING = True
NUM_CLASSES = data_tool.NUM_CLASSES  ## 为空位+1，classes最后一位(classes[26])代表空位
if IS_TRAINING:  ## 训练时的参数
    ## CNN
    P_KEEP_CONV = 0.9  ## 卷积层1中Dropout
    P_KEEP_BILSTM = 0.7
    ## FC
    P_KEEP_FC = 0.8  ## FC中dropout
    IS_SUMMARY = True  ## 是否进行在Tensorboard上显示网络中数据，graphs还是会显示的。
else:  #### 测试集时参数
    ## CNN
    P_KEEP_CONV = 1  ## 测试集时，不要用Dropout
    P_KEEP_BILSTM = 1
    ## FC
    P_KEEP_FC = 1
    IS_SUMMARY = False

## CNN layer
CONV_CORE_NUM_1 = 3  ## 卷积层1,卷积核个数
CONV_CORE_SIZE_1 = [3, 3, 3, CONV_CORE_NUM_1]  ## 卷积层1,卷积核大小
CONV_STRIDES_1 = [1, 1, 1, 1]  ## 卷积层1,卷积步长，中间两位为准:1x1
CONV_PADDING_1 = "SAME"  ## 卷积层1,卷积padding,SAME不改变h,w
POOL_CORE_SIZE_1 = [1, 1, 1, 1]  ## 池化核大小，中间两位为准，2x2
POOL_STRIDES_1 = [1, 1, 1, 1]  ## 池化，步长，中间两位为准，2x2
POOL_PADDING_1 = "SAME"  ## 池化padding类型
## MDLSTM layer
MDLSTM_HIDDEN_NUM_UNITS_1 = 100  ## lstm中神经单元个数
MDLSTM_HIDDEN_NUM_UNITS_2 = 80
MDLSTM_HIDDEN_NUM_UNITS_3 = 50
MDLSTM_FC_NUM_OUTPUTS_1 = 6  ## mdlstm前会进行fully_connected，fc的output单元个数，output.shape=[?,h,w,num_output]
MDLSTM_FC_NUM_OUTPUTS_2 = 30
MDLSTM_FC_NUM_OUTPUTS_3 = 50
## BiLSTM
BILSTM_HIDDEN_NUM_UNITS_1 = 125
## FC layer
FC_NUM_NUITS_1 = 100  ## 神经单元个数
## 正则化regularizer
IS_REGULARIZER = False  ## 是否开启L2正则化
REGULARIZATION_RATE = 1e-3  ## 正则化系数


def get_weight( name, shape):
    initializer = tf.truncated_normal_initializer(stddev=0.1, mean=0)
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

def cnn_network(inputs):  # beta1.0 :  image->Conv->Relu->Maxpool->cnn_output
    shape = inputs.get_shape().as_list()
    input_h, input_w = shape[1], shape[2]
    print("cnn_network--inputs.shape=", shape," ,in training:",IS_TRAINING)
    # tf.summary.image("cnn_input_image", inputs, 5)  # 用于Tensorboard可视化显示，
    with tf.variable_scope("Conv_MaxPool_layer1"):  ##  CC{大小:3*3*3*8,步长:1,SAME} + PC{2*2,2x2,SAME}
        conv1_w = get_weight("conv1_w", CONV_CORE_SIZE_1)
        conv1_b = get_bias("conv1_b", [CONV_CORE_NUM_1])
        # 卷积
        conv1_preactivation = tf.nn.bias_add(
            tf.nn.conv2d(inputs, conv1_w, strides=CONV_STRIDES_1, padding=CONV_PADDING_1, name="Conv1"),
            conv1_b)
        conv1_a = tf.nn.relu(conv1_preactivation)
        # MaxPool1
        conv1_p = tf.nn.max_pool(conv1_a, ksize=POOL_CORE_SIZE_1, strides=POOL_STRIDES_1,
                                 padding=POOL_PADDING_1, name="MaxPool1")
        # Dropout
        conv1 = tf.nn.dropout(conv1_p, P_KEEP_CONV)
        ## Conv_MaxPool_layer1.output.shape=[?, h/2, w/2, CONV_CORE_NUM_1]
        print("cnn_network--conv1.shape=", conv1.shape)

    if IS_SUMMARY:
        tf.summary.image("cnn_input_image", inputs, 5)  # 用于Tensorboard可视化显示，
        tf.summary.histogram("Conv1/weights", conv1_w)
        tf.summary.histogram("Conv1/biases", conv1_b)
        # Tensorboard-用于可视化Conv-MaxPool后的图像
        conv1_for_show1 = tf.reshape(conv1_a[:, :, :, 0], [-1, input_h, input_w, 1])
        conv1_for_show2 = tf.reshape(conv1_a[:, :, :, 1], [-1, input_h, input_w, 1])
        conv1_for_show3 = tf.reshape(conv1_a[:, :, :, 2], [-1, input_h, input_w, 1])
        tf.summary.image("conv1_for_show1", conv1_for_show1, 5)  # 输出带有5张图像的summary协议缓冲区，channel1
        tf.summary.image("conv1_for_show2", conv1_for_show2, 5)  # 输出带有5张图像的summary协议缓冲区，channel2
        tf.summary.image("conv1_for_show3", conv1_for_show3, 5)  # 输出带有5张图像的summary协议缓冲区，channel3
    return conv1

def mdlstm_network(inputs):  # beta1.0 :  cnn_output->FC->MDLSTM->FC->MDLSTM->FC->MDLSTM->mdlstm_output
    shape = inputs.get_shape().as_list()
    input_h, input_w = shape[1], shape[2]  ## 这个img_h,img_W不是真实image大小，因为经过了卷积与池化
    print("mdlstm_network--inputs.shape=", shape)
    ## 确定MDLSTM输出out_box大小[height,width]，
    output_box_1_size = [input_h, input_w]
    output_box_2_size = [input_h, input_w]
    output_box_3_size = [input_h, input_w]
    img_size = [input_h, input_w]
    ## 通过out_box反推mdlstm需要的window，例如：如果要mdlstm输入输出height,width大小不变，那window=[1,1]
    window_size_1 = data_tool.window_size_calculator(img_size, output_box_1_size)
    window_size_2 = data_tool.window_size_calculator(output_box_1_size, output_box_2_size)
    window_size_3 = data_tool.window_size_calculator(output_box_2_size, output_box_3_size)

    with tf.name_scope("FC-MDLSTM_layer1"):
        ## FC全连接层: [?,h,w,c] -> [?,h,w, FC_NUM_OUTPUTS_1]
        fc1_output = tf.contrib.slim.fully_connected(inputs=inputs, num_outputs=MDLSTM_FC_NUM_OUTPUTS_1,
                                                     activation_fn=tf.tanh)
        print("mdlstm_network--fc1_output.shape=", fc1_output.shape)

        ## MDLSTM:[?,h,w,c] -> [?,h/win_h,w/win_c, MDLSTM_HIDDEN_NUM_UNITS_1]
        mdlstm1_output, _ = md_lstm.multi_dimensional_rnn_while_loop(rnn_size=MDLSTM_HIDDEN_NUM_UNITS_1,
                                                                     input_data=fc1_output, sh=window_size_1,
                                                                     scope_n="MDLSTM_layer1")
        print("mdlstm_network--mdlstm1_output.shape=", mdlstm1_output.shape)

    with tf.name_scope("FC-MDLSTM_layer2"):
        ## FC全连接层: [?,h,w,c] -> [?,h,w, FC_NUM_OUTPUTS_2]
        fc2_output = tf.contrib.slim.fully_connected(inputs=mdlstm1_output,
                                                     num_outputs=MDLSTM_FC_NUM_OUTPUTS_2,
                                                     activation_fn=tf.tanh)
        print("mdlstm_network--fc2_output.shape=", fc2_output.shape)
        ## MDLSTM:[?,h,w,c] -> [?,h/win_h,w/win_c, MDLSTM_HIDDEN_NUM_UNITS_2]
        mdlstm2_output, _ = md_lstm.multi_dimensional_rnn_while_loop(rnn_size=MDLSTM_HIDDEN_NUM_UNITS_2,
                                                                     input_data=fc2_output, sh=window_size_2,
                                                                     scope_n="MDLST_layer2")
        print("mdlstm_network--mdlstm2_output.shape=", mdlstm2_output.shape)

    with tf.name_scope("FC-MDLSTM_layer3"):
        ## FC全连接层: [?,h,w,c] -> [?,h,w, FC_NUM_OUTPUTS_3]
        fc3_output = tf.contrib.slim.fully_connected(inputs=mdlstm2_output,
                                                     num_outputs=MDLSTM_FC_NUM_OUTPUTS_3,
                                                     activation_fn=tf.tanh)
        print("mdlstm_network--fc3_output.shape=", fc3_output.shape)
        ## MDLSTM:[?,h,w,c] -> [?,h/win_h,w/win_c, MDLSTM_HIDDEN_NUM_UNITS_3]
        mdlstm3_output, _ = md_lstm.multi_dimensional_rnn_while_loop(rnn_size=MDLSTM_HIDDEN_NUM_UNITS_3,
                                                                     input_data=fc3_output, sh=window_size_3,
                                                                     scope_n="MDLSTM_layer3")
        print("mdlstm_network--mdlstm3_output.shape=", mdlstm3_output.shape)

    # Tensorboard
    if IS_SUMMARY:
        tf.summary.tensor_summary("FC-MDLSTM_layer1/mdlstm1_output", mdlstm1_output)  # 输出一个序列化的协议缓冲区
        tf.summary.tensor_summary("FC-MDLSTM_layer2/mdlstm2_output", mdlstm2_output)  # 输出一个序列化的协议缓冲区
        tf.summary.tensor_summary("FC-MDLSTM_layer3/mdlstm3_output", mdlstm3_output)  # 输出一个序列化的协议缓冲区
        ## mdlstm3_output.shape=[?,h,w,MDLSTM_HIDDEN_NUM_UNITS_3]
        # Tensorboard-用于可视化mdlstm后的图像
        mdlstm_for_show1 = tf.expand_dims(mdlstm3_output[:, :, :, 1], axis=3)
        mdlstm_for_show2 = tf.expand_dims(mdlstm3_output[:, :, :, 2], axis=3)
        mdlstm_for_show3 = tf.expand_dims(mdlstm3_output[:, :, :, 3], axis=3)
        tf.summary.image("mdlstm_for_show1", mdlstm_for_show1, 5)  # 输出带有5张图像的summary协议缓冲区，channel1
        tf.summary.image("mdlstm_for_show2", mdlstm_for_show2, 5)  # 输出带有5张图像的summary协议缓冲区，channel2
        tf.summary.image("mdlstm_for_show3", mdlstm_for_show3, 5)  # 输出带有5张图像的summary协议缓冲区，channel3

    ## [?,h,w,MDLSTM_HIDDEN_NUM_UNITS_3]  -> [?,w,MDLSTM_HIDDEN_NUM_UNITS_3]
    mdlstm_layer_output = tf.reduce_sum(mdlstm3_output, axis=1)
    print("mdlstm_network--mdlstm_layer_output.shape=", mdlstm_layer_output.shape)
    return mdlstm_layer_output  # [?,h,w,classes]


def bi_lstmm_network(inputs):
    shape = inputs.get_shape().as_list()    # [batch, height, width, features]
    input_h, input_w = shape[1], shape[2]  ## 这个img_h,img_W不是真实image大小，因为经过了卷积与池化
    print("bi_lstmm_network--inputs.shape=", shape)
    with tf.variable_scope('Reshaping_cnn'):
        transposed = tf.transpose(inputs, perm=[0, 2, 1, 3],
                                  name='transpose')  # [? , width, height, features]
        conv_reshaped = tf.reshape(transposed, [-1, shape[2], shape[1] * shape[3]],
                                   name='reshape')  # [? , width, height x features]

    list_n_hidden = [BILSTM_HIDDEN_NUM_UNITS_1, BILSTM_HIDDEN_NUM_UNITS_1]

    with tf.name_scope("BiLSTM_layer1"):
        # Forward direction cells
        fw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]
        # Backward direction cells
        bw_cell_list = [BasicLSTMCell(nh, forget_bias=1.0) for nh in list_n_hidden]

        bilstm1_a, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list,
                                                                        bw_cell_list,
                                                                        conv_reshaped,
                                                                        dtype=tf.float32
                                                                        )
        # Dropout layer
        bilstm1_output = tf.nn.dropout(bilstm1_a, keep_prob=P_KEEP_BILSTM)
        # [? , width, 2*BILSTM_HIDDEN_NUM_UNITS_1]

        with tf.variable_scope('FC_layer1'):
            shape = bilstm1_output.get_shape().as_list()  # [?, width, 2*BILSTM_HIDDEN_NUM_UNITS_1]
            fc1_output = slim.layers.linear(bilstm1_output, NUM_CLASSES)  # [batch x width, n_class]

            fc1_output = tf.reshape(fc1_output, [-1, shape[1], NUM_CLASSES],
                                  name='fc1_out')  # [batch, width, n_classes]

            # Swap batch and time axis for ctc loss
            logprob = tf.transpose(fc1_output, [1, 0, 2], name='transpose_time_major')  # [width(time), batch, n_classes]

    print("bi_lstmm_network--output.shape=", logprob.get_shape().as_list())
    return logprob


def fc_network(inputs):  # beta1.0 :  mdlstm_output->FC->fc_output
    shape = inputs.get_shape().as_list()
    batch_size = shape[0]
    print("fc_network--inputs.shape=", shape)
    with tf.variable_scope("FC_layer1"):
        ## [batch_size,max_time_step,BILSTM_HIDDEN_NUM_UNITS_1]->reshape->[batch_size*max_time_step,num_hidden]
        fc_inputs = tf.reshape(inputs, [-1, MDLSTM_HIDDEN_NUM_UNITS_3])

        fc1_weight = get_weight(name="fc1_weight", shape=[MDLSTM_HIDDEN_NUM_UNITS_3, FC_NUM_NUITS_1])
        fc1_bias = get_bias(name="fc1_bias", shape=[FC_NUM_NUITS_1])
        ## [batch_size*max_time_step,num_hidden]-> A*W+b->[batch_size*max_time_step,fc_num_unit_1]
        fc1_a = tf.nn.relu(tf.matmul(fc_inputs, fc1_weight) + fc1_bias, name="fc1_relu")
        fc1_output = tf.nn.dropout(fc1_a, P_KEEP_FC)
        print("fc_network---fc1_output.shape=", fc1_output.shape)
    with tf.variable_scope("FC_layer2"):
        fc2_weight = get_weight(name="fc2_weight", shape=[FC_NUM_NUITS_1, NUM_CLASSES])
        fc2_bias = get_bias(name="fc2_bias", shape=[NUM_CLASSES])
        ## [batch_size*max_time_step,fc_num_unit_1]-> A*W+b->[batch_size*max_time_step,num_classes]
        fc2_output = tf.matmul(fc1_output, fc2_weight) + fc2_bias
        ## [batch_size*max_time_step,num_classes]->reshape-> [batch_size, max_time_step, num_classes]
        fc2_output = tf.reshape(fc2_output, [batch_size, -1, NUM_CLASSES])
        ## [batch_size,max_time_step,num_classes]->transpose -> [max_time_step,batch_size,num_classes]
        fc2_output = tf.transpose(fc2_output, (1, 0, 2))

    # Tensorboard
    if IS_SUMMARY:
        # Tensorboard
        tf.summary.tensor_summary("FC_layer1/fc1_output", fc1_output)  # 输出一个序列化的协议缓冲区
        tf.summary.histogram("FC_layer1/fc1_weight", fc1_weight)
        tf.summary.histogram("FC_layer1/fc1_bias", fc1_bias)
        # Tensorboard
        tf.summary.tensor_summary("FC_layer2/fc2_output", fc2_output)  # 输出一个序列化的协议缓冲区
        tf.summary.histogram("FC_layer2/fc2_weight", fc2_weight)
        tf.summary.histogram("FC_layer2/fc2_bias", fc2_bias)
    print("fc_network---fc2_output.shape=", fc2_output.shape, ",batch_size=", batch_size)
    return fc2_output

