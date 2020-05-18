from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import io, transform

import trainer.utils.voc_classname_encoder as voc_classname_encoder

import trainer.model.CenterNet as net
import trainer.utils.tfrecord_voc_utils as voc_utils
import trainer.model.Centernet_New as centernet_new

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# lr = 0.001
# batch_size = 15
# buffer_size = 256
# epochs = 160
# epochs = 2
reduce_lr_epoch = []


def train(args):
    inputsize = args.input_size

    config = {
        'backbone': 'hourglass', # hourglass, dla
        'mode': args.mode,  # 'train', 'test'
        'input_size': inputsize,
        'data_format': args.data_format,  # 'channels_last' 'channels_first'
        'num_classes': args.num_classes,
        'weight_decay': 1e-4,
        'keep_prob': 0.5,  # not used
        'batch_size': args.batch_size,
        'score_threshold': 0.1,
        'top_k_results_output': 100,
    }

    image_augmentor_config = {
        'data_format': args.data_format,
        'output_shape': [inputsize, inputsize],
        'zoom_size': [600, 600],
        'crop_method': 'random',
        'flip_prob': [0., 0.5],
        'fill_mode': 'BILINEAR',
        'keep_aspect_ratios': False,
        'constant_values': 0.,
        'color_jitter_prob': 0.5,
        'rotate': [0.5, -5., -5.],
        'pad_truth_to': 60,
    }

    data = os.listdir(args.train_files)
    data = [os.path.join(args.train_files, name) for name in data]

    train_gen = voc_utils.get_generator(data, args.batch_size, args.buffer_size, image_augmentor_config)
    trainset_provider = {
        'data_shape': [inputsize, inputsize, 3],
        'num_train': 26,
        'num_val': 0,  # not used
        'train_generator': train_gen,
        'val_generator': None  # not used
    }
    # centernet = net.CenterNet(config, trainset_provider)
    centernet = centernet_new.CenterNetNew(config, trainset_provider)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    lr = 0.001
    # your code
    for i in range(args.num_epochs):
        start_time = time.time()
        print('-' * 25, 'epoch', i, '-' * 25)
        if i in reduce_lr_epoch:
            lr = lr / 10.
            print('reduce lr, lr=', lr, 'now')
        mean_loss = centernet.train_one_epoch(lr)
        print('>> mean loss', mean_loss)
        centernet.save_weight('latest', args.job_dir)  # 'latest', 'best
        elapsed_time = time.time() - start_time
        print('Duration: ', str(elapsed_time))

def write_graph(session, logsdir):
    graph = session.graph
    tf.train.write_graph(graph, logsdir, 'centernet.pbtxt', True)
    train_writer = tf.summary.FileWriter(logsdir)
    train_writer.add_graph(graph)
    train_writer.flush()
    train_writer.close()

def test(args):
    input_size = args.input_size
    weights_path = args.weights_path
    num_classes = args.num_classes
    data_format = args.data_format
    batch_size = args.batch_size
    logs_dir = args.logs_dir

    config = {
        'mode': 'test',
        'input_size': input_size,
        'data_format': data_format,
        'num_classes': num_classes,
        'batch_size': batch_size,
        'score_threshold': 0.1,
        'top_k_results_output': 100,

    }
    # centernet = net.CenterNet(config, None)
    centernet =  centernet_new.CenterNetNew(config, None)
    centernet.load_pretrained_weight(weights_path)
    write_graph(centernet.sess, logs_dir)

    img = io.imread('000048.jpg')
    img = transform.resize(img, [input_size, input_size])
    img = np.expand_dims(img, 0)
    result = centernet.test_one_image(img)
    id_to_clasname = {k: v for (v, k) in voc_classname_encoder.classname_to_ids.items()}
    scores = result[0]
    bbox = result[1]
    class_id = result[2]
    print(scores, bbox, class_id)
    plt.figure(1)
    plt.imshow(np.squeeze(img))
    axis = plt.gca()
    for i in range(len(scores)):
        if(class_id[i] == 14):
            rect = patches.Rectangle((bbox[i][1], bbox[i][0]), bbox[i][3] - bbox[i][1], bbox[i][2] - bbox[i][0],
                                 linewidth=2, edgecolor='b', facecolor='none')
            axis.add_patch(rect)
            plt.text(bbox[i][1], bbox[i][0], id_to_clasname[class_id[i]] + str(' ') + str(scores[i]), color='red',
                 fontsize=12)
    plt.show()

    print('done')


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    # Input Arguments

    # execution config
    PARSER.add_argument('--train-files', help='GCS file or local paths to training data', default='./data')
    PARSER.add_argument('--job-dir', help='GCS location to write checkpoints and export models',default='./centernet/test')
    PARSER.add_argument('--num-epochs', type=int, default=2)
    PARSER.add_argument('--mode', choices=['train', 'test'], default='train')
    PARSER.add_argument('--weights-path', default='../centernet/test-5') # ../test_remote/run_05152020_0900/test-6719
    PARSER.add_argument('--logs-dir', default='../logs/') # ../logs2/

    # model config
    PARSER.add_argument('--data-format', choices=['channels_last', 'channels_first'], default='channels_last')
    PARSER.add_argument('--num-classes', type=int, default=20)
    PARSER.add_argument('--batch-size', type=int, default=15)
    PARSER.add_argument('--buffer-size', type=int, default=256)
    PARSER.add_argument('--input-size', type=int, default=512)

    ARGUMENTS, _ = PARSER.parse_known_args()

    tf.logging.set_verbosity('INFO')
    # Suppress C++ level warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Run the training job
    train(ARGUMENTS)
    # test(ARGUMENTS)