from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import trainer.utils.tfrecord_voc_utils as voc_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tfrecord = voc_utils.dataset2tfrecord('./VOC2007/Annotations', './VOC2007/JPEGImages',
                                      './data/', 'test', 10)
print(tfrecord)
