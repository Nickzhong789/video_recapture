import os
import cv2
import numpy as np
import pandas as pd
import operator


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


def video_cap(video_label_list):
    [video, v_label, v_idx] = video_label_list
    video_obj = cv2.VideoCapture(video)
    ret_val = 1
    count = 0
    frame_list = []
    while ret_val:
        ret_val, image = video_obj.read()
        count += 1
        if count % 5 == 0:
            frame_list.append([image, v_label])
        if count == 15:
            break

    frame_block_list = []
    f_idx = 0
    for frame, label in frame_list[:-1]:
        block_list = []
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
        for in_block in input_blocks[:3]:
            op_list = []
            for i in range(16):
                k_size = (3, 3) if i < 8 else (5, 5)
                op = cv2.GaussianBlur(in_block, k_size, sigmaX_list[i % 8])
                op_list.append(op)
            block_list.append([np.stack(op_list), v_label, b_idx])
        frame_block_list.append(block_list)
        f_idx += 1

    return frame_block_list


def get_blocks(df_v):
    video_info_list = []
    for i in range(len(df_v)):
        frame_list = video_cap(df_v.iloc[i])
        for block_list in frame_list:
            for block_info in block_list:
                video_info_list.append(block_info)

    return video_info_list
