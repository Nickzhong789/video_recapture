import os
import cv2
import numpy as np
import pandas as pd
import operator


def generate_anno(path):
    video_list = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            fname = os.path.join(root, name)
            file_cls = fname.split('/')[2]
            label = 0 if file_cls == "0" else 1
            video_list.append([fname, label])

    df = pd.DataFrame(data=video_list, columns=['video', 'label'])
    df.to_csv('./video_anno.csv', index=None)


def video_cap(v_l_list):
    [video, v_label] = v_l_list
    video_obj = cv2.VideoCapture(video)
    ret_val = 1
    count = 0
    frame_list = []
    while ret_val:
        ret_val, image = video_obj.read()
        count += 1
        if count % 5 == 0:
            frame_list.append([image, v_label])

    f_b_list = []
    for frame, label in frame_list:
        block_label_list = []
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
        op_list = []
        for in_block in input_blocks:
            for j in range(16):
                k_size = (3, 3) if i < 8 else (5, 5)
                op = cv2.GaussianBlur(in_block, k_size, sigmaX_list[j % 8])
                op_list.append(op)
            block_label_list.append([np.stack(op_list), v_label])

        f_b_list.append(block_label_list)

    return f_b_list


def split_all(path):
    all_parts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            all_parts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            all_parts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            all_parts.insert(0, parts[1])
    return all_parts


def Label_Encode(block_labels_training):
    values = np.array(block_labels_training)

    block_labels_training_encoded = [0] * len(block_labels_training)

    for i in range(0, len(block_labels_training)):
        if block_labels_training[i] == 'original':
            block_labels_training_encoded[i] = 0
            return block_labels_training_encoded[i]
        else:
            block_labels_training_encoded[i] = 1
            return block_labels_training_encoded[i]


def Label_Encode_Valid(block_labels_valid):
    block_labels_valid_encode = []
    values = np.array(block_labels_valid)

    for i in range(0, len(values)):
        if values[i] == 'recaptured':
            block_labels_valid_encode.append(1)
        else:
            block_labels_valid_encode.append(0)

    return block_labels_valid_encode


if __name__ == '__main__':
    root_path = './video_data/'
    generate_anno(root_path)

    video_info_list = []
    df_v = pd.read_csv('./video_anno.csv')
    for i in range(len(df_v)):
        video_label_list = df_v.iloc[i]

        frame_block_list = video_cap(video_label_list)
        video_info_list.append(frame_block_list)

    print(video_info_list)
