import tensorflow as tf
from keras.layers import (Layer, Dense, BatchNormalization, Conv1D, Dropout, Reshape, Add, LeakyReLU, 
                          MaxPooling1D, Activation, MultiHeadAttention, LayerNormalization, Flatten)
from keras import Model, Sequential, Input, regularizers
import numpy as np

seed = 42
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)
initializer = tf.keras.initializers.GlorotUniform(seed)
# params = {
#     'regularizer': 0.25,
# }
deacy_rate = 0.001

def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.15)

class ConvolutionBlock(Layer):
    def __init__(self, n_filter, n_kernel=3):
        super(ConvolutionBlock, self).__init__()
        # Convolution Block
        self.conv1 = Conv1D(filters=n_filter, 
                            kernel_size=n_kernel, 
                            padding='same',
                            activation=leaky_relu, 
                            kernel_initializer=initializer, 
                            kernel_regularizer=regularizers.l2(deacy_rate),
                            bias_initializer=initializer)
        self.conv2 = Conv1D(filters=n_filter, 
                            kernel_size=n_kernel, 
                            padding='same',
                            activation=leaky_relu, 
                            kernel_initializer=initializer, 
                            kernel_regularizer=regularizers.l2(deacy_rate),
                            bias_initializer=initializer)
        self.conv3 = Conv1D(filters=n_filter, 
                            kernel_size=n_kernel, 
                            padding='same',
                            activation=leaky_relu, 
                            kernel_initializer=initializer, 
                            kernel_regularizer=regularizers.l2(deacy_rate),
                            bias_initializer=initializer)
        self.pool = MaxPooling1D(pool_size=2, strides=2)
        # Input Transfer
        self.conv_res = Conv1D(filters=n_filter, 
                               kernel_size=1, 
                               padding='same',
                               activation=leaky_relu, 
                               kernel_initializer=initializer, 
                               kernel_regularizer=regularizers.l2(deacy_rate),
                               bias_initializer=initializer)
        self.pool_res = MaxPooling1D(pool_size=2, strides=2)
        # Residual Connect
        self.add = Add()

    def call(self, inputs):
        # Convolution Block
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.pool(outputs)
        # Input Transfer
        out_res = self.conv_res(inputs)
        out_res = self.pool_res(out_res)
        # Residual Connect
        outputs = self.add([outputs, out_res])
        return outputs


class DilatedConvBlock(Layer):
    def __init__(self):
        super(DilatedConvBlock, self).__init__()
        self.dila_conv1 = Conv1D(filters=128, 
                                 kernel_size=7, 
                                 dilation_rate=1, 
                                 padding='same',
                                 activation=leaky_relu, 
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizers.l2(deacy_rate),
                                 bias_initializer=initializer)
        self.dila_conv2 = Conv1D(filters=128, 
                                 kernel_size=7, 
                                 dilation_rate=2, 
                                 padding='same',
                                 activation=leaky_relu, 
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizers.l2(deacy_rate),
                                 bias_initializer=initializer)
        self.dila_conv3 = Conv1D(filters=128, 
                                 kernel_size=7, 
                                 dilation_rate=4, 
                                 padding='same',
                                 activation=leaky_relu, 
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizers.l2(deacy_rate),
                                 bias_initializer=initializer)
        self.dila_conv4 = Conv1D(filters=128, 
                                 kernel_size=7, 
                                 dilation_rate=8, 
                                 padding='same',
                                 activation=leaky_relu, 
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizers.l2(deacy_rate),
                                 bias_initializer=initializer)
        self.dila_conv5 = Conv1D(filters=128, 
                                 kernel_size=7, 
                                 dilation_rate=16, 
                                 padding='same',
                                 activation=leaky_relu, 
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizers.l2(deacy_rate),
                                 bias_initializer=initializer)
        self.dila_conv6 = Conv1D(filters=128, 
                                 kernel_size=7, 
                                 dilation_rate=32, 
                                 padding='same',
                                 activation=leaky_relu, 
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizers.l2(deacy_rate),
                                 bias_initializer=initializer)
        self.dropout = Dropout(0.2)
        self.add = Add()

    def call(self, inputs, training):
        outputs = self.dila_conv1(inputs)
        outputs = self.dila_conv2(outputs)
        outputs = self.dila_conv3(outputs)
        outputs = self.dila_conv4(outputs)
        outputs = self.dila_conv5(outputs)
        outputs = self.dila_conv6(outputs)
        outputs = self.dropout(outputs, training=training)
        outputs = self.add([outputs, inputs])
        return outputs

""" ref: https://tensorflow.google.cn/tutorials/text/transformer?hl=zh_cn """
def scaled_dot_product_attention(q, k, v):
    """计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。


    参数:
        q: 请求的形状 == (..., seq_len_q, depth)
        k: 主键的形状 == (..., seq_len_k, depth)
        v: 数值的形状 == (..., seq_len_v, depth_v)


    返回值:
        输出，注意力权重
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # 缩放 matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # softmax 在最后一个轴（seq_len_k）上归一化，因此分数相加等于1。
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, dtype='float32')
        self.wk = tf.keras.layers.Dense(d_model, dtype='float32')
        self.wv = tf.keras.layers.Dense(d_model, dtype='float32')

        self.dense = tf.keras.layers.Dense(d_model, dtype='float32')

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
    

class PositionalEncoding(Layer):
    def __init__(self, length, depth):
        super(PositionalEncoding, self).__init__()
        self.length = length
        self.depth = depth

    def get_config(self):
        config = super().get_config()
        config.update({
            "length": self.length,
            "depth": self.depth,
        })
        return config

    @staticmethod
    def positional_encoding(length, depth):
        depth = depth/2

        positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

        angle_rates = 1 / (10000**depths)         # (1, depth)
        angle_rads = positions * angle_rates      # (pos, depth)

        pos_encoding = np.concatenate(
          [np.sin(angle_rads), np.cos(angle_rads)],
          axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        tensor_dtype = inputs.dtype
        outputs = inputs + tf.cast(self.positional_encoding(self.length, self.depth), tensor_dtype)
        return outputs
    
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class TransformerEncoder(Layer):
    def __init__(self, n_channel, n_length, n_head, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.pos_coding = PositionalEncoding(n_channel, n_length)
        self.mha = MultiHeadAttention(d_model=n_length,
                                      num_heads=n_head)
        self.ffn = point_wise_feed_forward_network(n_length, n_length*4)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output, _ = self.mha(inputs, inputs, inputs)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class FeatureShortRange(Layer):
    def __init__(self):
        super(FeatureShortRange, self).__init__()
        self.conv_block0 = ConvolutionBlock(16)
        self.conv_block1 = ConvolutionBlock(16)
        self.conv_block2 = ConvolutionBlock(32)
        self.conv_block3 = ConvolutionBlock(32)
        self.conv_block4 = ConvolutionBlock(64)
        self.conv_block5 = ConvolutionBlock(64)
        self.conv_block6 = ConvolutionBlock(128)
        self.conv_block7 = ConvolutionBlock(128)
        self.conv_block8 = ConvolutionBlock(256)
        self.conv_block9 = ConvolutionBlock(256)
        # Window, Dense
        self.window = Reshape(target_shape=(256,))

    def call(self, inputs):
        # Feature
        shape_0 = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs, [shape_0 * 1200, 1024, 1])
        outputs = self.conv_block0(inputs)
        outputs = self.conv_block1(outputs)
        outputs = self.conv_block2(outputs)
        outputs = self.conv_block3(outputs)
        outputs = self.conv_block4(outputs)
        outputs = self.conv_block5(outputs)
        outputs = self.conv_block6(outputs)
        outputs = self.conv_block7(outputs)
        outputs = self.conv_block8(outputs)
        outputs = self.conv_block9(outputs)
        # Window, Dense
        outputs = self.window(outputs)
        outputs = tf.reshape(outputs, [shape_0, 1200, 256])
        return outputs
    

class FeatureLongRange(Layer):
    def __init__(self, n_kernel):
        super(FeatureLongRange, self).__init__()
        self.n_kernel = n_kernel
        self.conv_block0 = ConvolutionBlock(16, n_kernel)
        self.conv_block1 = ConvolutionBlock(16, n_kernel)
        self.conv_block2 = ConvolutionBlock(32, n_kernel)
        self.conv_block3 = ConvolutionBlock(32, n_kernel)
        self.conv_block4 = ConvolutionBlock(64, n_kernel)
        self.conv_block5 = ConvolutionBlock(64, n_kernel)
        self.conv_block6 = ConvolutionBlock(128, n_kernel)
        self.conv_block7 = ConvolutionBlock(128, n_kernel)
        self.conv_block8 = ConvolutionBlock(256, n_kernel)
        self.conv_block9 = ConvolutionBlock(256, n_kernel)
        # Window, Dense
        self.window = Reshape(target_shape=(1200, 256))

    def call(self, inputs):
        # Feature
        outputs = self.conv_block0(inputs)
        outputs = self.conv_block1(outputs)
        outputs = self.conv_block2(outputs)
        outputs = self.conv_block3(outputs)
        outputs = self.conv_block4(outputs)
        outputs = self.conv_block5(outputs)
        outputs = self.conv_block6(outputs)
        outputs = self.conv_block7(outputs)
        outputs = self.conv_block8(outputs)
        outputs = self.conv_block9(outputs)
        # Window, Dense
        outputs = self.window(outputs)
        return outputs

    
class StackedConv(Layer):
    def __init__(self, filters, rate, n_kernel=9):
        super(StackedConv, self).__init__()
        self.dila_conv1 = Conv1D(filters=filters, 
                                 kernel_size=n_kernel, 
                                 dilation_rate=rate, 
                                 padding='same',
                                 activation=leaky_relu, 
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizers.l2(deacy_rate),
                                 bias_initializer=initializer)
        self.dila_conv2 = Conv1D(filters=filters, 
                                 kernel_size=n_kernel, 
                                 dilation_rate=2*rate, 
                                 padding='same',
                                 activation=leaky_relu, 
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizers.l2(deacy_rate),
                                 bias_initializer=initializer)
        self.dila_conv3 = Conv1D(filters=filters, 
                                 kernel_size=n_kernel, 
                                 dilation_rate=4*rate, 
                                 padding='same',
                                 activation=leaky_relu, 
                                 kernel_initializer=initializer,
                                 kernel_regularizer=regularizers.l2(deacy_rate),
                                 bias_initializer=initializer)
        self.batchnormal = BatchNormalization()
        self.maxpool = MaxPooling1D(pool_size=2, strides=2, padding='same')
        self.dropout = Dropout(0.2)

    def call(self, x):
        x = self.dila_conv1(x)
        x = self.dila_conv2(x)
        x = self.dila_conv3(x)
        x = self.batchnormal(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x
    
    
class AHIConvModel(Layer):
    def __init__(self):
        super(AHIConvModel, self).__init__()
        
        self.conv_block1 = StackedConv(filters=64, rate=1)
        self.conv_block2 = StackedConv(filters=64, rate=2)
        self.conv_block3 = StackedConv(filters=32, rate=3)
        self.conv_block4 = StackedConv(filters=16, rate=4)
        self.flatten = Flatten()
        self.fc = Dense(256, dtype='float32')

    def call(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    

class AHIPredModel(Layer):
    def __init__(self, rate=0.2):
        super(AHIPredModel, self).__init__()
        self.fc1 = Dense(units=64, dtype='float32')
        self.batch_norm1 = BatchNormalization()
        self.leaky_relu1 = LeakyReLU(alpha=0.15)
        self.dropout1 = Dropout(rate)
        self.fc2 = Dense(units=32, dtype='float32')
        self.batch_norm2 = BatchNormalization()
        self.leaky_relu2 = LeakyReLU(alpha=0.15)
        self.dropout2 = Dropout(rate)
        self.fc3 = Dense(units=16, dtype='float32')
        self.batch_norm3 = BatchNormalization()
        self.leaky_relu3 = LeakyReLU(alpha=0.15)
        self.dropout3 = Dropout(rate)
        self.fc_out = Dense(units=1, dtype='float32')  # Estimated AHI

    def call(self, inputs, training):
        x = self.fc1(inputs)
        x = self.batch_norm1(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x, training=training)
        x = self.fc3(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu3(x)
        x = self.dropout3(x, training=training)
        # Final output
        return self.fc_out(x)

    
class DomainPred(Layer):
    def __init__(self):
        super(DomainPred, self).__init__()
        self.conv1 = Conv1D(filters=64, 
                            kernel_size=3, 
                            padding='same',
                            activation=leaky_relu, 
                            kernel_initializer=initializer, 
                            kernel_regularizer=regularizers.l2(deacy_rate),
                            bias_initializer=initializer)
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling1D(pool_size=2, strides=2, padding='same')
        
        self.conv2 = Conv1D(filters=32, 
                            kernel_size=3, 
                            padding='same',
                            activation=leaky_relu, 
                            kernel_initializer=initializer, 
                            kernel_regularizer=regularizers.l2(deacy_rate),
                            bias_initializer=initializer)
        self.bn2 = BatchNormalization()
        self.pool2 = MaxPooling1D(pool_size=2, strides=2, padding='same')
        
        self.conv3 = Conv1D(filters=16, 
                            kernel_size=3, 
                            padding='same',
                            activation=leaky_relu, 
                            kernel_initializer=initializer, 
                            kernel_regularizer=regularizers.l2(deacy_rate),
                            bias_initializer=initializer)
        self.bn3 = BatchNormalization()
        self.pool3 = MaxPooling1D(pool_size=2, strides=2, padding='same')
        
        self.conv4 = Conv1D(filters=8, 
                            kernel_size=3, 
                            padding='same',
                            activation=leaky_relu, 
                            kernel_initializer=initializer, 
                            kernel_regularizer=regularizers.l2(deacy_rate),
                            bias_initializer=initializer)
        self.bn4 = BatchNormalization()
        self.pool4 = MaxPooling1D(pool_size=2, strides=2, padding='same')
        
        self.flat = Flatten()
        
        self.fc1 = Dense(units=128, 
                         activation=leaky_relu, 
                         dtype='float32')
        self.bn5 = BatchNormalization()

        self.fc2 = Dense(units=64, 
                         activation=leaky_relu,
                         dtype='float32')
        self.bn6 = BatchNormalization()
        
        self.fc3 = Dense(units=5, 
                         activation='softmax',
                         dtype='float32')


    def call(self, inputs):
        # Convolution Block
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.pool1(outputs)
        
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = self.pool2(outputs)
        
        outputs = self.conv3(outputs)
        outputs = self.bn3(outputs)
        outputs = self.pool3(outputs)
        
        outputs = self.conv4(outputs)
        outputs = self.bn4(outputs)
        outputs = self.pool4(outputs)
        
        outputs = self.flat(outputs)
        outputs = self.fc1(outputs)
        outputs = self.bn5(outputs)
        outputs = self.fc2(outputs)
        outputs = self.bn6(outputs)
        outputs = self.fc3(outputs)
        
        return outputs
    

class Sleep_NET(Model):
    def __init__(self, n_kernel):
        super(Sleep_NET, self).__init__()
        
        # Long-Range Feature, Input_Shape = [batch_size, 1200*1024, 1], Output_Shape = [batch_size, 1200, 10s24]
        self.long_range_fea = FeatureLongRange(n_kernel=n_kernel)

        # Short-Range Feature, Input_Shape = [batch_size*1200, 1024, 1], Output_Shape = [batch_size, 1200, 1024]
        self.short_range_fea = FeatureShortRange()

        # Dense Layer
        self.dense = Dense(128, dtype='float32')

        # Dilated_Block
        self.dila_block1 = DilatedConvBlock()
        self.mha_block1 = TransformerEncoder(1200, 128, 4)
        self.dila_block2 = DilatedConvBlock()
        self.mha_block2 = TransformerEncoder(1200, 128, 4)
        
        # Apnea Severity Prediction
        self.ahiconv = AHIConvModel()
        self.ahipred = AHIPredModel()

        # Sleep Prediction
        self.conv_pred = Conv1D(filters=4, kernel_size=1)
        self.activate = Activation('softmax', dtype='float32')
        
        # Domain Adaption
        self.domain = DomainPred()

        
    def call(self, inputs, training):
        """ Generic Features """
        # long-range
        outputs1 = self.long_range_fea(inputs)
        # short-range
        outputs = self.short_range_fea(inputs)
        outputs = tf.concat([outputs, outputs1], axis=2)
        # Dense
        outputs = self.dense(outputs)
        
        """ Domain Adaption """
        outputs_domain = self.domain(outputs)
        
        """ AHI Predict """
        # outputs_ahi = self.ahiconv(outputs[:, :840, :])
        outputs_ahi = self.ahiconv(outputs)
        outputs_ahi = self.ahipred(outputs_ahi, training=training)

        """ Sleep Staging """
        # Dilated_Block
        outputs = self.dila_block1(outputs, training=training)
        outputs = self.mha_block1(outputs, training=training)
        outputs = self.dila_block2(outputs, training=training)
        outputs = self.mha_block2(outputs, training=training)
        # Prediction
        outputs = self.conv_pred(outputs)
        outputs = self.activate(outputs)

        return outputs, outputs_ahi, outputs_domain
