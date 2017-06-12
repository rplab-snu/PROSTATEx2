import tensorflow as tf
from ops import acts, layers, losses, utils

class VGG_net(object):
    def __init__(self):
        self.act_fn = tf.nn.relu
        self.num_class = 5
        self.kernel_num = 16
        self.pool_kernel = 2

    def inference(self, input_):
        conv1 = layers.conv2d_same_repeat(input_, self.kernel_num, num_repeat=2, activation_fn=self.act_fn, name="down1")
        pool1 = layers.max_pool(conv1, name="pool1")

        conv2 = layers.conv2d_same_repeat(pool1, self.kernel_num * 2, num_repeat=2, activation_fn=self.act_fn, name="down2")
        pool2 = layers.max_pool(conv2, name="pool2")

        conv3 = layers.conv2d_same_repeat(pool2, self.kernel_num * 4, num_repeat=3, activation_fn=self.act_fn, name="down3")
        pool3 = layers.max_pool(conv3, name="pool3")

        conv4 = layers.conv2d_same_repeat(pool3, self.kernel_num * 8, num_repeat=3, activation_fn=self.act_fn, name="down4")
        pool4 = layers.max_pool(conv4, name="pool4")

        conv5 = layers.conv2d_same_repeat(pool4, self.kernel_num * 8, num_repeat=3, activation_fn=self.act_fn, name="down5")
        pool5 = layers.global_avg_pool(conv5, name="pool5")

        conv6 = layers.bottleneck_act(pool5, self.kernel_num * 8, activation_fn=self.act_fn, name="down6")
        conv7 = layers.bottleneck_layer(conv6, self.num_class, name="down7")

        logits = layers.flatten(conv7, 'flat')

        return logits


class ResNet(object):
    def __init__(self):
        self.act_fn = tf.nn.relu
        self.num_class = 5
        self.num_kernel = 64
        self.pool_kernel = 2
        self.stride_down = 2
        self.layer_def = [3, 6, 3]  # setting res-net 34

    def shortcut(self, input_, identity, name='shortcut'):
        with tf.variable_scope(name):
            if input_.get_shape() == identity.get_shape():
                # f(x) + I(x)
                sc = tf.add(input_, identity, name='add')
            else:  # if input_.get_shape() != identity.get_shape():
                output_dim = input_.get_shape().as_list()[-1]
                stride = int(identity.get_shape().as_list()[1] / input_.get_shape().as_list()[1] + 0.5)
                # down-sampling with strided 1x1 conv
                conv = layers.bottleneck_layer(identity, output_dim, d_h=stride, d_w=stride, name='conv')
                bn = layers.batch_norm(conv, name='bn')
                # f(x) + down(I(x))
                sc = tf.add(input_, bn, name='add')

            return sc

    """ input -> bn -> act -> conv -> bn -> act -> conv -> shortcut """

    def basicBlock(self, input_, output_dim, stride=1, name='basic'):
        assert output_dim is not 0

        with tf.variable_scope(name):
            input_dim = input_.get_shape().as_list()[-1]

            bn = layers.batch_norm(input_, name='bn')
            act = self.act_fn(bn, name='act')

            conv1 = layers.conv2d_same_act(act, input_dim, activation_fn=self.act_fn, name='conv_1')
            conv2 = layers.conv2d_same(conv1, output_dim, d_h=stride, d_w=stride, name='conv_2')
            # make sure that identity is input_ and output of this block is conv2
            sc = self.shortcut(conv2, input_, name='short')

            return sc

    def layer(self, input_, output_dim, repeat_num, stride=1, name='blocks'):
        assert repeat_num is not 0

        with tf.variable_scope(name):
            input_dim = input_.get_shape().as_list()[-1]

            for i in range(repeat_num - 1):
                block = self.basicBlock(input_, input_dim, name='block_%d' % (i + 1))
                input_ = block

            block = self.basicBlock(input_, output_dim, stride, name='block_%d' % repeat_num)

            return block

    def layer_repeat(self, input_, layer_config, name='layers'):
        assert len(layer_config) is not 0

        with tf.variable_scope(name):
            num_layers = len(layer_config)

            layer = 0
            for i in range(num_layers):
                layer = self.layer(input_, self.num_kernel * pow(2, i), layer_config[i],
                                   self.stride_down, name='layer_%d' % (i + 1))
                input_ = layer

            return layer

    def inference(self, input_, reuse=False):
        with tf.variable_scope('ResNet') as scope:
            if reuse:
                scope.reuse_variables()

            conv1 = layers.conv2d_same_act(input_, self.num_kernel, k_h=7, k_w=7, d_h=2, d_w=2,
                                           activation_fn=self.act_fn, name='conv_1')

            pool1 = layers.max_pool(conv1, k_h=self.pool_kernel, k_w=self.pool_kernel,
                                    padding='SAME', name='pool1')

            layer_blocks = self.layer_repeat(pool1, self.layer_def, name='layers')

            pool2 = layers.global_avg_pool(layer_blocks, name='pool2')

            flat = layers.flatten(pool2, 'flat')

            linear = layers.linear(flat, self.num_class, name='linear')

            logit = tf.sigmoid(linear, name='logit')

            return logit


