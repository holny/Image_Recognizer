import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow.python.ops.rnn import dynamic_rnn


def ln(tensor, scope=None, epsilon=1e-5):
    """ Layer normalizes a 2D tensor along its second axis """

    assert (len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    ln_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return ln_initial * scale + shift


class MultiDimensionalLSTMCell(RNNCell):
    # RNNCell class inheritance

    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, forget_bias=0.0, activation=tf.nn.tanh):
        self._num_units = num_units
        # hidden unit의 수

        self._forget_bias = forget_bias
        # forget_bias    ->    for forget gate

        self._activation = activation 
        # activation -> sigmoid로 변경
        # activation default -> hyperbolic tangent function

        #constructor

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)
    # state_size -> LSTM의 초기 state들이 set 됨
    # num_unit -> 16이므로, 16

    # property -> getter, setter처럼 private 변수의 직접 접근을 막기 위해 사용됨

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):

        # __call__ -> 함수처럼 class를 실행하게 할 수 있음.
        #           ex) helloworld = HelloWorld()       -> class instantiation
        #               helloworld()                    -> class를 function처럼 실행

        """Long short-term memory cell (LSTM).
        @param: inputs (batch,n)
        @param state: the states and hidden unit of the two cells
        """
        with tf.variable_scope(scope or type(self).__name__):
            c1, c2, h1, h2 = state

            # change bias argument to False since LN will add bias via shift
            concat = _linear([inputs, h1, h2], 5 * self._num_units, False)

            i, j, f1, f2, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

            # add layer normalization to each gate
            #i = ln(i, scope='i/')
            #j = ln(j, scope='j/')
            #f1 = ln(f1, scope='f1/')
            #f2 = ln(f2, scope='f2/')
            #o = ln(o, scope='o/')

            new_c = (c1 * tf.nn.sigmoid(f1 + self._forget_bias) +
                     c2 * tf.nn.sigmoid(f2 + self._forget_bias) + tf.nn.sigmoid(i) *
                     self._activation(j))

            #new_c = (c1 * f1 + c2 * f2 + i * self._activation(j))

            # add layer_normalization in calculation of new hidden state
            new_h = self._activation(ln(new_c, scope='new_h/')) * tf.nn.sigmoid(o)
            #new_h = self._activation(ln(new_c, scope='new_h/')) * o
            new_state = LSTMStateTuple(new_c, new_h)

            return new_h, new_state


def multi_dimensional_rnn_while_loop(rnn_size, input_data, sh, dims=None, scope_n="layer1"):
    """Implements naive multi dimension recurrent neural networks

    @param rnn_size: the hidden units
    @param input_data: the data to process of shape [batch,h,w,channels]
    @param sh: [height,width] of the windows
    @param dims: dimensions to reverse the input data,eg.
        dims=[False,True,True,False] => true means reverse dimension
    @param scope_n : the scope


    returns [batch,h/sh[0],w/sh[1],rnn_size] the output of the lstm
        batch = 1, h/sh[0] = 28 / 2, w/sh[1] = 28 / 2, rnn_size = 16
            -> [1, 14, 14, 16] -> [batch_size, h_step, w_step, rnn_size]
    """

    with tf.variable_scope("MultiDimensionalLSTMCell-" + scope_n, initializer = tf.contrib.layers.xavier_initializer(uniform=False)):

        # Create multidimensional cell with selected size
        cell = MultiDimensionalLSTMCell(rnn_size)
        # rnn size: hidden size (16)
        # instance creation

        # Get the shape of the input (batch_size, x, y, channels)
        shape = input_data.get_shape().as_list()
        # tensor의 shape를 list로 만들어 반환
        # 4D -> batch, height, width, channels

        batch_size = shape[0]
        X_dim = shape[1]
        Y_dim = shape[2]
        channels = shape[3]

        # Window size
        X_win = sh[0]
        Y_win = sh[1]
        # Get the runtime batch size
        batch_size_runtime = tf.shape(input_data)[0]
        # tensor의 shape을 return         ->      [0]이므로 batch_size가 실시간으로 return

        # If the input cannot be exactly sampled by the window, we patch it with zeros
        if X_dim % X_win != 0:
            # Get offset size
            offset = tf.zeros([batch_size_runtime, X_win - (X_dim % X_win), Y_dim, channels])
            # Concatenate X dimension
            # offset = 1 - (28 % 1)  =>  1, Y_Dim -> 1

            input_data = tf.concat(axis=1, values=[input_data, offset])
            # Get new shape

            shape = input_data.get_shape().as_list()
            # Update shape value
            X_dim = shape[1]

        # The same but for Y axis
        if Y_dim % Y_win != 0:
            # Get offset size
            offset = tf.zeros([batch_size_runtime, X_dim, Y_win - (Y_dim % Y_win), channels])
            # Concatenate Y dimension
            input_data = tf.concat(axis=2, values=[input_data, offset])
            # Get new shape
            shape = input_data.get_shape().as_list()
            # Update shape value
            Y_dim = shape[2]

        # Get the steps to perform in X and Y axis
        h_steps, w_steps = int(X_dim / X_win), int(Y_dim / Y_win)

        # Get the number of features (total number of input values per step)
        features = Y_win * X_win * channels
                # 4(width) * 3(height) * 3(channel) = 36

        # Reshape input data to a tensor containing the step indexes and features inputs
        # The batch size is inferred from the tensor size
        x = tf.reshape(input_data, [batch_size_runtime, h_steps, w_steps, features])
      
        # Reverse the selected dimensions
        if dims is not None:
        #    assert dims[0] is False and dims[3] is False
            x = tf.reverse(x, dims)

        # Reorder inputs to (h, w, batch_size, features)
        x = tf.transpose(x, [1, 2, 0, 3])
        # Reshape to a one dimensional tensor of (h*w*batch_size , features)

        x = tf.reshape(x, [-1, features])
        # ? x 1 size로 reshape

        # Split tensor into h*w tensors of size (batch_size , features)
        x = tf.split(axis=0, num_or_size_splits=h_steps * w_steps, value=x)


        # Create an input tensor array (literally an array of tensors) to use inside the loop
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=h_steps * w_steps, name='input_ta')
        # Unstack the input X in the tensor array
        inputs_ta = inputs_ta.unstack(x)
        # Create an input tensor array for the states
        states_ta = tf.TensorArray(dtype=tf.float32, size=h_steps * w_steps + 1, name='state_ta', clear_after_read=False)
        # And an other for the output
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=h_steps * w_steps, name='output_ta')

        # initial cell hidden states
        # Write to the last position of the array, the LSTMStateTuple filled with zeros
        states_ta = states_ta.write(h_steps * w_steps, LSTMStateTuple(tf.zeros([batch_size_runtime, rnn_size], tf.float32),
                                                        tf.zeros([batch_size_runtime, rnn_size], tf.float32)))


        # rnn_size = 6, batch_size_runtime = 40
        # 784, LSTMStateTuple([40, 6], zeros(40, 6)

        # Function to get the sample skipping one row
        def get_up(t_, w_):
            return t_ - tf.constant(w_)

        # Function to get the previous sample
        def get_last(t_, w_):
            return t_ - tf.constant(1)

        # Controls the initial index
        time = tf.constant(0)
        zero = tf.constant(0)

        # Body of the while loop operation that applies the MD LSTM
        def body(time_, outputs_ta_, states_ta_):
            # If the current position is less or equal than the width, we are in the first row
            # and we need to read the zero state we added in row (h*w). 
            # If not, get the sample located at a width distance.
            state_up = tf.cond(tf.less_equal(time_, tf.constant(w_steps)),
                               lambda: states_ta_.read(h_steps * w_steps),
                               lambda: states_ta_.read(get_up(time_, w_steps)))

            # If it is the first step we read the zero state if not we read the immediate last
            state_last = tf.cond(tf.less(zero, tf.mod(time_, tf.constant(w_steps))),
                                 lambda: states_ta_.read(get_last(time_, w_steps)),
                                 lambda: states_ta_.read(h_steps * w_steps))
            print("multi_dimensional_rnn_while_loop--state_last.shape=",state_last.get_shape())

            # We build the input state in both dimensions
            current_state = state_up[0], state_last[0], state_up[1], state_last[1]
            # print("multi_dimensional_rnn_while_loop--state_up[0]=",state_up[0]," ,state_last[0]=", state_last[0], " ,state_up[1]=",state_up[1], " ,state_last[1]=", state_last[1])
            # print("multi_dimensional_rnn_while_loop--current_state=",current_state)
            # Now we calculate the output state and the cell output
            out, state = cell(inputs_ta.read(time_), current_state)
            # We write the output to the output tensor array
            outputs_ta_ = outputs_ta_.write(time_, out)
            # And save the output state to the state tensor array
            states_ta_ = states_ta_.write(time_, state)

            # Return outputs and incremented time step 
            return time_ + 1, outputs_ta_, states_ta_

        # Loop output condition. The index, given by the time, should be less than the
        # total number of steps defined within the image
        def condition(time_, outputs_ta_, states_ta_):
            return tf.less(time_, tf.constant(h_steps * w_steps))

        # Run the looped operation

        result, outputs_ta, states_ta = tf.while_loop(condition, body, [time, outputs_ta, states_ta],
                                                      parallel_iterations= 1)
        
        # Extract the output tensors from the processesed tensor array
        outputs = outputs_ta.stack()
        states = states_ta.stack()

        # Reshape outputs to match the shape of the input
        y = tf.reshape(outputs, [h_steps, w_steps, batch_size_runtime, rnn_size])

        # Reorder te dimensions to match the input
        y = tf.transpose(y, [2, 0, 1, 3])
        # Reverse if selected
        if dims is not None:
            y = tf.reverse(y, dims)

        # Return the output and the inner states
        return y, states


def horizontal_standard_lstm(input_data, rnn_size):
    # input is (b, h, w, c)
    b, h, w, c = input_data.get_shape().as_list()
    # transpose = swap h and w.
    new_input_data = tf.reshape(input_data, (b * h, w, c))  # horizontal.
    rnn_out, _ = dynamic_rnn(tf.contrib.rnn.LSTMCell(rnn_size),
                             inputs=new_input_data,
                             dtype=tf.float32)
    rnn_out = tf.reshape(rnn_out, (b, h, w, rnn_size))
    return rnn_out


def snake_standard_lstm(input_data, rnn_size):
    # input is (b, h, w, c)
    b, h, w, c = input_data.get_shape().as_list()
    # transpose = swap h and w.
    new_input_data = tf.reshape(input_data, (b, w * h, c))  # snake.
    rnn_out, _ = dynamic_rnn(tf.contrib.rnn.LSTMCell(rnn_size),
                             inputs=new_input_data,
                             dtype=tf.float32)
    rnn_out = tf.reshape(rnn_out, (b, h, w, rnn_size))
    return rnn_out
