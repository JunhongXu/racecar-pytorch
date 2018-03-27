import numpy as np
import io
from collections import defaultdict
from configparser import ConfigParser
from yolo import YOLO
import torch
from model import yolo_cfg as y


def unique_config_sections(config_file):
    """Convert all config sections to have unique names.
    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def build_yolo(cfg_filepath, weight_path):
    tiny_yolo = YOLO(y)
    state_dict = tiny_yolo.state_dict()
    keys = list(state_dict.keys())
    stream = unique_config_sections(cfg_filepath)
    weights = open(weight_path, 'rb')

    # header has 16 bytes
    weights.read(16)

    cfg = ConfigParser()
    cfg.read_file(stream)
    cfg_dict = {}

    # total size of the networks in mb
    total_size = 0

    # total number of parameters
    total_params = 0

    # keep track of weights
    weight_num = 0

    for section in cfg.sections():
        # network configurations
        if section.startswith('net'):
            for key in cfg[section]:
                if key in ['height', 'width', 'channels']:
                    cfg_dict[key] = int(cfg[section][key])

        # read convolutional layer
        if section.startswith('convolutional'):
            # TODO: PARAMS SHOULD BE SAVED AS A DICTIONARY
            nfilters = int(cfg[section]['filters'])
            ksize = int(cfg[section]['size'])
            activation = cfg[section]['activation']
            stride = int(cfg[section]['stride'])
            batch_norm = 'batch_normalize' in cfg[section]
            pad = int(cfg[section]['pad'])

            conv_bias_size = nfilters * 4



            conv_bias = np.frombuffer(weights.read(conv_bias_size), dtype=np.float32)

            if batch_norm:
                # weight, [], running_mean, running_variance
                bn_weights = np.frombuffer(weights.read(nfilters * 3 * 4), dtype=np.float32).reshape(3, -1)
                bn_nparam = np.prod(bn_weights.shape)

                total_params += nfilters * 3
                total_size += bn_nparam * 4

                bn_weights = np.asarray([
                    bn_weights[0],
                    conv_bias,
                    bn_weights[1],
                    bn_weights[2]
                ])

                state_dict[keys[weight_num + 1]].copy_(torch.from_numpy(bn_weights[0]))
                state_dict[keys[weight_num + 2]].copy_(torch.from_numpy(conv_bias))
                state_dict[keys[weight_num + 3]].copy_(torch.from_numpy(bn_weights[1]))
                state_dict[keys[weight_num + 4]].copy_(torch.from_numpy(bn_weights[2]))

            if section == 'convolutional_20':
                prev_nfilter = 512
            elif section == 'convolutional_21':
                prev_nfilter = 1280
            elif section == 'convolutional_0':
                prev_nfilter = 3

            conv_weights_shape = (nfilters, prev_nfilter, ksize, ksize)
            conv_weight_nparams = np.prod(conv_weights_shape)
            conv_weight_size = conv_weight_nparams * 4

            conv_weights = np.frombuffer(weights.read(conv_weight_size), dtype=np.float32).reshape(*conv_weights_shape)

            total_size += conv_bias_size + conv_weight_size
            total_params += nfilters + conv_weight_nparams

            state_dict[keys[weight_num]].copy_(torch.from_numpy(conv_weights))

            if not batch_norm:
                state_dict[keys[weight_num + 1]].copy_(torch.from_numpy(conv_bias))
                weight_num += 2
            else:
                weight_num += 5
            print(conv_weights.shape, conv_bias.shape, bn_weights.shape if batch_norm else ' ' )

            prev_nfilter = nfilters

        #TODO: LAYERS SHOULD BE SAVED AS DICTIONARY
        if section.startswith('maxpool'):
            pass

        if section.startswith('region'):
            pass

    remaining_weights = weights.read()
    print('Number of remaining weights: %i, should be zero' % (len(remaining_weights)/4))
    print('Total parameters, %.4f million\nTotal size of the model %.4f mbs' %(total_params/1000000, total_size/1000000))
    torch.save(tiny_yolo.state_dict(), 'tiny-yolo.pth')


build_yolo('../model/yolo.cfg', '../model/yolo.weights')


