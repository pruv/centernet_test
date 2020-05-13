import tensorflow as tf


class DLABackbone:

    def __init__(self, data_format, is_training):
        self.data_format = data_format
        self.is_training = is_training

    def build_model(self, images):
        with tf.variable_scope('backone'):
            conv = self._conv_bn_activation(
                bottom=images,
                filters=16,
                kernel_size=7,
                strides=1,
            )
            conv = self._conv_bn_activation(
                bottom=conv,
                filters=16,
                kernel_size=3,
                strides=1,
            )
            conv = self._conv_bn_activation(
                bottom=conv,
                filters=32,
                kernel_size=3,
                strides=2,
            )
            dla_stage3 = self._dla_generator(conv, 64, 1, self._basic_block)
            dla_stage3 = self._max_pooling(dla_stage3, 2, 2)

            dla_stage4 = self._dla_generator(dla_stage3, 128, 2, self._basic_block)
            residual = self._conv_bn_activation(dla_stage3, 128, 1, 1)
            residual = self._avg_pooling(residual, 2, 2)
            dla_stage4 = self._max_pooling(dla_stage4, 2, 2)
            dla_stage4 = dla_stage4 + residual

            dla_stage5 = self._dla_generator(dla_stage4, 256, 2, self._basic_block)
            residual = self._conv_bn_activation(dla_stage4, 256, 1, 1)
            residual = self._avg_pooling(residual, 2, 2)
            dla_stage5 = self._max_pooling(dla_stage5, 2, 2)
            dla_stage5 = dla_stage5 + residual

            dla_stage6 = self._dla_generator(dla_stage5, 512, 1, self._basic_block)
            residual = self._conv_bn_activation(dla_stage5, 512, 1, 1)
            residual = self._avg_pooling(residual, 2, 2)
            dla_stage6 = self._max_pooling(dla_stage6, 2, 2)
            dla_stage6 = dla_stage6 + residual
        with tf.variable_scope('upsampling'):
            dla_stage6 = self._conv_bn_activation(dla_stage6, 256, 1, 1)
            dla_stage6_5 = self._dconv_bn_activation(dla_stage6, 256, 4, 2)
            dla_stage6_4 = self._dconv_bn_activation(dla_stage6_5, 256, 4, 2)
            dla_stage6_3 = self._dconv_bn_activation(dla_stage6_4, 256, 4, 2)

            dla_stage5 = self._conv_bn_activation(dla_stage5, 256, 1, 1)
            dla_stage5_4 = self._conv_bn_activation(dla_stage5 + dla_stage6_5, 256, 3, 1)
            dla_stage5_4 = self._dconv_bn_activation(dla_stage5_4, 256, 4, 2)
            dla_stage5_3 = self._dconv_bn_activation(dla_stage5_4, 256, 4, 2)

            dla_stage4 = self._conv_bn_activation(dla_stage4, 256, 1, 1)
            dla_stage4_3 = self._conv_bn_activation(dla_stage4 + dla_stage5_4 + dla_stage6_4, 256, 3, 1)
            dla_stage4_3 = self._dconv_bn_activation(dla_stage4_3, 256, 4, 2)

            features = self._conv_bn_activation(dla_stage6_3 + dla_stage5_3 + dla_stage4_3, 256, 3, 1)
            features = self._conv_bn_activation(features, 256, 1, 1)
        self.features = features

    def _dconv_bn_activation(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu):
        conv = tf.layers.conv2d_transpose(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
        )
        bn = self._bn(conv)
        if activation is not None:
            bn = activation(bn)
        return bn

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
        return bn

    def _dla_generator(self, bottom, filters, levels, stack_block_fn):
        if levels == 1:
            block1 = stack_block_fn(bottom, filters)
            block2 = stack_block_fn(block1, filters)
            aggregation = block1 + block2
            aggregation = self._conv_bn_activation(aggregation, filters, 3, 1)
        else:
            block1 = self._dla_generator(bottom, filters, levels - 1, stack_block_fn)
            block2 = self._dla_generator(block1, filters, levels - 1, stack_block_fn)
            aggregation = block1 + block2
            aggregation = self._conv_bn_activation(aggregation, filters, 3, 1)
        return aggregation

    def _conv_bn_activation(self, bottom, filters, kernel_size, strides, activation=tf.nn.relu):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format
        )
        bn = self._bn(conv)
        if activation is not None:
            return activation(bn)
        else:
            return bn

    def _basic_block(self, bottom, filters):
        conv = self._conv_bn_activation(bottom, filters, 3, 1)
        conv = self._conv_bn_activation(conv, filters, 3, 1)
        axis = 3 if self.data_format == 'channels_last' else 1
        input_channels = tf.shape(bottom)[axis]
        shutcut = tf.cond(
            tf.equal(input_channels, filters),
            lambda: bottom,
            lambda: self._conv_bn_activation(bottom, filters, 1, 1)
        )
        return conv + shutcut

    def _max_pooling(self, bottom, pool_size, strides, name=None):
        return tf.layers.max_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _avg_pooling(self, bottom, pool_size, strides, name=None):
        return tf.layers.average_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )
