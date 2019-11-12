# coding:utf-8
import tensorflow as tf
import data_tool as data_tool,inference as inference
import time
import os


class HW_Recognizer():
    def __init__(self, is_training=True):
        inference.IS_TRAINING = is_training
        self.IS_TRAINING = is_training  ## True:使用Train set， False:使用Test set，并不进行反向
        ## decay learning rate  指数衰减学习率
        self.START_LEARNING_RATE = 0.001  ## 初始学习率
        self.DECAY_STEPS = 100  ## DECAY_STEPS
        self.DECAY_RATE = 0.96  ## DECAY_RATE
        self.IS_STAIRCASE = False  ## 是否阶梯下降
        ## 正则化regularizer
        self.IS_REGULARIZER = inference.IS_REGULARIZER  ## 是否开启L2正则化
        self.REGULARIZATION_RATE = inference.REGULARIZATION_RATE  ## 正则化系数
        if self.IS_TRAINING:  ## 训练时的参数
            self.EPOCHS = data_tool.TRAIN_EPOCHS  ## epoch
            self.BATCH_SIZE = data_tool.TRAIN_BATCH_SIZE  ## batch_size
            self.IS_NEED_SAVE = True  ## 是否保存训练模型数据
            self.MODEL_PATH = "./ModelTrain"  ## 模型数据保存地址
            self.LOG_PATH = "./SummaryTrain"  ## Tensorboard Log保存地址
        else:  #### 测试集时参数
            self.EPOCHS = data_tool.TEST_EPOCHS
            self.BATCH_SIZE = data_tool.TEST_BATCH_SIZE  ## 对测试集进行验证model，batch_size设置为整个Test数据集大小，epoch=1
            self.IS_NEED_SAVE = False
            self.MODEL_PATH = "./ModelTrain"
            self.LOG_PATH = "./SummaryTest"

    def compute_cost(self, labels_sparse, inputs, sequence_length):
        with tf.name_scope("CTC_loss"):
            loss = tf.nn.ctc_loss(labels=labels_sparse, inputs=inputs, sequence_length=sequence_length,
                                  ignore_longer_outputs_than_inputs=True)
            cost = tf.reduce_mean(loss)
            tf.summary.scalar("ctc_loss", cost)
        return cost

    def get_prediction(self, inputs, sequence_length):
        with tf.name_scope("Beam_search"):
            ## 前面说的划分块之后找每块的类属概率分布，ctc_beam_search_decoder方法,是每次找最大的K个概率分布
            ## 还有一种贪心策略是只找概率最大那个，也就是K=1的情况ctc_ greedy_decoder
            decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs=inputs, sequence_length=sequence_length)
        ## decoded[0]只是预测出的sparseTensor
        return decoded[0]

    def compute_distance(self, pred_sparse, labels_sparse):
        with tf.name_scope("Compute_distance"):
            distance = tf.reduce_mean(tf.edit_distance(tf.cast(pred_sparse, tf.int32), labels_sparse))
            # Tensorborad
            tf.summary.scalar("distance", distance)
        return distance

    def train(self):
        images, labels_seq, _, seq_lens = data_tool.get_data_from_TFrecord(is_training=self.IS_TRAINING)
        ## tf.nn.ctc_loss()中Labels需要传入sparseTensor。
        # 从TFrecord获取label sparse的indices，然后values已知都是1，shape已知=[chars_max_num,num_classes]，可以直接得出sparse
        labels_sparse = tf.sparse_placeholder(dtype=tf.int32, name='labels_sparse')
        global_steps = tf.Variable(0, trainable=False, name="global_steps", dtype=tf.int32)
        ## inference
        cnn_ouput = inference.cnn_network(images)
        md_lstm_output = inference.mdlstm_network(cnn_ouput)
        logistics = inference.fc_network(md_lstm_output)
        # ctc_loss
        cost = self.compute_cost(labels_sparse=labels_sparse, inputs=logistics, sequence_length=seq_lens)
        # 正则化
        if self.IS_REGULARIZER:
            # L2正则化--2/2--创建变量时，tensorflow会将变量加入集合 tf.GraphKeys.REGULARIZATOIN_LOSSES，
            regularization_cost = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            cost = cost + regularization_cost

        if self.IS_TRAINING:  ## 如果正在测试集上验证就不进行反向传播
            # 指数衰减学习率
            learning_rate = tf.train.exponential_decay(self.START_LEARNING_RATE, global_steps, self.DECAY_STEPS,
                                                       self.DECAY_RATE, staircase=self.IS_STAIRCASE)
            # optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_steps)

        # 通过beamSearch获取预测结果
        pred_sparse = self.get_prediction(inputs=logistics, sequence_length=seq_lens)
        prediction = tf.sparse_tensor_to_dense(tf.cast(pred_sparse, tf.int32))
        # 计算预测与真实之间的距离
        distance = self.compute_distance(pred_sparse, labels_sparse)

        # Saver
        if self.IS_NEED_SAVE and (not os.path.exists(self.MODEL_PATH)):
            os.makedirs(self.MODEL_PATH)
        saver = tf.train.Saver()
        # Tensorboard
        merged = tf.summary.merge_all()

        # for test code
        labels = tf.sparse_tensor_to_dense(labels_sparse)

        with tf.Session() as sess:
            print("Train init......") if self.IS_TRAINING else print("Test init......")
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())  ## 有队列迭代读取数据就要初始化本地变量
            ckpt = tf.train.get_checkpoint_state(self.MODEL_PATH)
            # Saver
            if self.IS_NEED_SAVE and ckpt and ckpt.model_checkpoint_path:
                # 注意Saver保存时设置的路径，路径不对读取不到。
                print("Saver-读取checkpoint,path=", ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                start = global_steps.eval()
                print("Saver-已成功读取Saver保存点，start from step:", start)
            # Tensorboard
            writer = tf.summary.FileWriter(self.LOG_PATH, sess.graph)
            # 队列化输入
            coord = tf.train.Coordinator()
            # tf.train.start_queue_runners会把graph里的所有队列run起来，并返回管理队列的对应的子线程
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            try:
                loop_batch_times = 1
                epoch = 1
                print("Train start......total_epochs:", self.EPOCHS, " ,start epoch:",
                      epoch) if self.IS_TRAINING else print("Test start......total_epochs:", self.EPOCHS,
                                                            " ,start epoch:", epoch)
                while not coord.should_stop():
                    start = time.time()
                    ## 从TFrecord中获取到label的sequence(长度定长为最大长度，不足padding。是数字序列,数字是字符在字典中的位置。)
                    ## sequence转为sparse,然后通过feed_dict传入placeholder
                    images_, labels_seq_ = sess.run([images, labels_seq])
                    label_sparse_feed = data_tool.labels_sequence_to_sparse(labels_seq_)
                    feed_dict = {labels_sparse: label_sparse_feed}

                    if self.IS_TRAINING:  ## 在训练集
                        if self.IS_REGULARIZER:  ## 开启了正则化就获取regularization_cost
                            learning_rate_,cost_, prediction_, regularization_cost_, _, global_steps_, distance_, merged_ = sess.run(
                                [learning_rate,cost, prediction, regularization_cost, optimizer, global_steps, distance, merged],
                                feed_dict=feed_dict)
                            print_regular_cost_str = " ,regular_cost=%d" % (regularization_cost_)
                        else:  ## 没开启正则化就不获取regularization_cost
                            learning_rate_,cost_, prediction_, _, global_steps_, distance_, merged_ = sess.run(
                                [learning_rate,cost, prediction, optimizer, global_steps, distance, merged],
                                feed_dict=feed_dict)
                            print_regular_cost_str = ""
                        print("In Training...times:%04d" % (loop_batch_times), " ,global_step:%06d" % (global_steps_),
                              " ,cost=", cost_, print_regular_cost_str, " ,distance=", distance_," ,learning_rate=",learning_rate_)

                    else:  ## 在测试集不进行反向optimizer
                        cost_, prediction_, global_steps_, distance_, merged_ = sess.run(
                            [cost, prediction, global_steps, distance, merged], feed_dict=feed_dict)
                        print("In testing...times:%04d" % (loop_batch_times), " ,global_step:%06d" % (global_steps_),
                              " ,cost=", cost_, " ,distance=", distance_)

                    ## Console show more
                    if (global_steps_ % 1 == 0):
                        for i in range(0, labels_seq_.shape[0], 100):
                            labels_seq_1 = labels_seq_[i]
                            prediction_1 = prediction_[i]
                            image_1 = images_[i]
                            label_str = data_tool.label_sequence_to_string(labels_seq_1)
                            pred_str = data_tool.label_sequence_to_string(prediction_1)
                            if self.IS_TRAINING:
                                print("In training...times:%04d" % (loop_batch_times), "Show more,  ,label=", label_str,
                                      " ,prediction=", pred_str)
                                ## for test
                                data_tool.show_one_sample_image(image_1, label_str)
                            else:
                                print("In testing...times:%04d" % (loop_batch_times), "Show more,   ,label=", label_str,
                                      " ,prediction=", pred_str)
                                ## for test
                                data_tool.show_one_sample_image(image_1, label_str)

                    ## Saver & Tensorboard
                    if (global_steps_ % 1 == 0):
                        # Tensorborad
                        writer.add_summary(merged_, global_step=global_steps_)
                        # Saver保存点
                        if self.IS_NEED_SAVE:
                            saver.save(sess, self.MODEL_PATH + "/model.ckpt", global_step=global_steps_)
                            print("Model has be saved on global_step=", global_steps_, " ,trained_times=",
                                  loop_batch_times)
                    loop_batch_times += 1

            except tf.errors.OutOfRangeError:
                print("Train done......") if self.IS_TRAINING else print("Test done......")
                if self.IS_NEED_SAVE:
                    saver.save(sess, self.MODEL_PATH + "/model.ckpt")
                    print("Final model has be saved！")
                coord.request_stop()

            finally:
                coord.request_stop()
                # 等待所有线程退出
            coord.join(threads)


nn = HW_Recognizer(True)
nn.train()
