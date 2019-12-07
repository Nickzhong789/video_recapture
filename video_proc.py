import os
import cv2
import numpy as np
import pandas as pd
import operator
from utils.osutil import *


def generate_anno(path):
    video_list = []
    idx = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            fname = os.path.join(root, name)
            file_cls = fname.split('/')[2]
            label = 0 if file_cls == "0" else 1
            video_list.append([fname, label, str(idx)])
            idx += 1

    df = pd.DataFrame(data=video_list, columns=['video', 'label', 'index'])
    df.to_csv('./video_anno.csv', index=None)


def video_cap(video_label_list, root_dir, train=True):  # transform a video to blocks
    [video, v_label, v_idx] = video_label_list
    data_dir = join(root_dir, str(v_idx))
    if not isdir(data_dir):
        mkdir_p(data_dir)

    video_obj = cv2.VideoCapture(video)
    ret_val = 1
    count = 0
    frame_list = []
    while ret_val:
        ret_val, image = video_obj.read()
        count += 1
        if count % 5 == 0:
            frame_list.append([image, v_label])

    f_idx = 0
    block_list = []
    for frame, label in frame_list[:-1]:
        gray = frame[:, :, 2]
        blocks = ([np.array(gray[m:m + 64, j:j + 64]) for j in range(0, gray.shape[1], 64)
                   for m in range(0, gray.shape[0], 64)])

        blocks_new = []
        variance_list = []
        for block in blocks:
            if block.shape[0] == 64 and block.shape[1] == 64:  # SELECTION OF ONLY 64x64 blocks
                blocks_new.append(block)
                variance_list.append(np.var(block))

        block_dict = {k: variance_list[k] for k in range(len(variance_list))}
        sorted_x = sorted(block_dict.items(), key=operator.itemgetter(1), reverse=True)

        input_blocks = []
        sorted_x = sorted_x if len(sorted_x) < 100 else sorted_x[:100]
        for x in sorted_x:
            idx = x[0]
            input_blocks.append(blocks_new[idx])

        sigmaX_list = [3.160, 3.800, 4.786, 6.309, 8.709, 12.589, 19.05, 31.62]
        b_idx = str(v_idx) + '_' + str(f_idx)
        frame_dir = join(data_dir, b_idx)
        print(frame_dir)
        if not isdir(frame_dir):
            mkdir_p(frame_dir)

        num = 0
        for in_block in input_blocks:
            op_list = []
            for i in range(16):
                k_size = (3, 3) if i < 8 else (5, 5)
                op = cv2.GaussianBlur(in_block, k_size, sigmaX_list[i % 8])
                op_list.append(op)
            block_dir = join(frame_dir, num_to_idx(num) + '.npy')
            np.save(block_dir, np.stack(op_list))
            is_train = 'train' if train is True else 'test'
            block_list.append([block_dir, str(v_label), b_idx, is_train])
            num += 1
        f_idx += 1

    return block_list


def num_to_idx(num):
    if num < 10:
        idx_str = '00' + str(num)
    elif num < 100:
        idx_str = '0' + str(num)
    else:
        idx_str = str(num)

    return idx_str
