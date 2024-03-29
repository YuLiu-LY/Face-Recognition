import os
import cv2
import math
import random
import numpy as np
import face_recognition as fr
from PIL import Image
from glob import glob
from tqdm import tqdm
from torchvision.transforms import Resize

DATA_ROOT = '/home/yuliu/Dataset/Face1'


def rotate(p, o, angle):
    '''
    Rotate point p around point o with angle
    p: (x, y)
    o: (x, y)
    angle: degree
    H: height of image
    '''
    angle = angle * np.pi / 180
    x = o[0] + math.cos(angle) * (p[0] - o[0]) - math.sin(angle) * (o[1] - p[1])
    y = o[1] - math.sin(angle) * (p[0] - o[0]) - math.cos(angle) * (o[1] - p[1])
    return int(x), int(y)


def align_face(face, landmarks):
    '''
    Align face by rotating face with eyes angle 
    face: (H, W, C)
    landmarks: dict
    '''
    eye_l = landmarks['left_eye']
    eye_r = landmarks['right_eye']
    # center of eyes
    center_l = np.mean(eye_l, axis=0)
    center_r = np.mean(eye_r, axis=0)
    center = (center_l + center_r) / 2
    center = center.astype(np.int32)
    center = (center[1].item(), center[0].item())
    # angle of eyes
    dist = center_r - center_l
    angle = np.arctan2(dist[1], dist[0]) * 180 / np.pi
    # rotate face to align eyes
    M_rotate = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_face = cv2.warpAffine(face, M_rotate, (face.shape[1], face.shape[0]))
    # rotate landmarks
    for key in landmarks.keys():
        for i in range(len(landmarks[key])):
            landmarks[key][i] = rotate(landmarks[key][i], center, angle)
    return rotated_face, landmarks


def check_valid_crop(top, bottom, left, right, H, W):
    top = max(0, top)
    bottom = min(H, bottom)
    left = max(0, left)
    right = min(W, right)
    return top, bottom, left, right


def crop_face(face, landmarks, img_size):
    '''
    Crop face by face landmarks
    face: (H, W, C)
    landmarks: dict, 68 points of face, (x, y)
    img_size: [H, W]
    The implementation here refers to the design of paper 'Pairwise Relational Networks for Face Recognition'
    '''
    H, W = img_size
    eye = [landmarks['left_eye'], landmarks['right_eye']]
    eye = np.concatenate(eye, axis=0)
    center_eye = eye.mean(axis=0).astype(np.int32)
    lip = [landmarks['top_lip'], landmarks['bottom_lip']]
    lip = np.concatenate(lip, axis=0)
    center_lip = lip.mean(axis=0).astype(np.int32)
    H_mid = center_lip[1] - center_eye[1] # the height of mid face (eye to lip), 35% of face height
    top = center_eye[1] - int(H_mid)
    bottom = center_lip[1] + int(H_mid / 0.35 * 0.3)
    d_y = bottom - top
    d_x = int(d_y * W / H)
    x_l = np.min(landmarks['chin'], axis=0)[0]
    x_r = np.max(landmarks['chin'], axis=0)[0]
    center_x = int((x_l + x_r) / 2)
    left = center_x - d_x // 2
    right = center_x + d_x // 2
    top, bottom, left, right = check_valid_crop(top, bottom, left, right, face.shape[0], face.shape[1])
    face = face[top:bottom, left:right]
    face = Resize(img_size)(Image.fromarray(face))
    return face


def get_face_img(path, model='hog', idx=0):
    img = fr.load_image_file(path)
    location = fr.face_locations(img, model=model)
    if len(location) == 0:
        return None
    landmarks = fr.face_landmarks(img, location)[idx] # 68 points of face, dict
    aligned_face, aligned_landmarks = align_face(img, landmarks)
    aligned_face = crop_face(aligned_face, aligned_landmarks, (112, 96))
    return aligned_face


def crop_and_align_all_face():
    # process train data
    print('Processing train data...')
    img_dirs = sorted(glob(f'{DATA_ROOT}/training_set/*'))
    img_paths_list = [sorted(glob(f'{img_dir}/*.jpg')) for img_dir in img_dirs]
    failed_list = []
    for img_paths in tqdm(img_paths_list):
        for img_path in img_paths:
            if img_path[-6:] == '_a.jpg' or os.path.exists(img_path.replace('.jpg', '_a.jpg')):
                continue
            aligned_face = get_face_img(img_path)
            if aligned_face is None:
                failed_list.append(img_path)
                continue
            # save img
            aligned_face.save(img_path.replace('.jpg', '_a.jpg'))
    # process failed data with cnn model
    print(f'Processing failed data: {len(failed_list)}...')
    for img_path in tqdm(failed_list):
        aligned_face = get_face_img(img_path, model='cnn')
        if aligned_face is None:
            continue
        # save img
        aligned_face.save(img_path.replace('.jpg', '_a.jpg'))
    print('Processing test data...')
    img_dirs = [f'{DATA_ROOT}/test_pair/{i}' for i in range(600)]
    img_paths_list = [[f'{dir}/A.jpg', f'{dir}/B.jpg'] for dir in img_dirs]
    for img_paths in tqdm(img_paths_list):
        for img_path in img_paths:
            if img_path[-6:] == '_a.jpg' or os.path.exists(img_path.replace('.jpg', '_a.jpg')):
                continue
            aligned_face = get_face_img(img_path, model='cnn')
            # save img
            aligned_face.save(img_path.replace('.jpg', '_a.jpg'))


def generate_data_set():
    # generate validation set
    img_dirs = sorted(glob(f'{DATA_ROOT}/training_set/*'))
    img_paths_list = [sorted(glob(f'{img_dir}/*_a.jpg')) for img_dir in img_dirs]
    # save train_val.txt
    with open(f'{DATA_ROOT}/train_val.txt', 'w') as f:
        for i in range(len(img_dirs)):
            if len(img_paths_list[i]) > 0:
                f.write(f'{img_dirs[idx]}\n')
    N = 300
    random.seed(0)
    idx_val = random.sample(range(len(img_dirs)), N)
    idx_train = [i for i in range(len(img_dirs)) if i not in idx_val]
    # save train.txt
    with open(f'{DATA_ROOT}/train.txt', 'w') as f:
        for idx in idx_train:
            if len(img_paths_list[idx]) > 0:
                f.write(f'{img_dirs[i]}\n')
    # save val.txt
    # generate pairs of same face and different face
    pair_list = []
    label_list = []
    N_pos = 0
    for i in idx_val:
        img_paths = img_paths_list[i]
        if len(img_paths) == 0:
            print(f'No face in {img_dirs[i]}')
            idx_val.pop(idx_val.index(i))
            continue
        if len(img_paths) == 2:
            pair_list.append(img_paths)
            label_list.append(1)
            N_pos += 1
        elif len(img_paths) > 2:
            for j in range(len(img_paths)):
                max_n = min(len(img_paths), 10)
                for k in range(j+1, max_n):
                    pair_list.append([img_paths[j], img_paths[k]])
                    label_list.append(1)
                    N_pos += 1
    N_neg = 0
    T = N_pos // len(idx_val)
    for i in idx_val:
        img_paths1 = img_paths_list[i]
        path1 = random.choice(img_paths1)
        for _ in range(T):
            j = random.choice(idx_val)
            while j == i:
                j = random.choice(idx_val)
            img_paths2 = img_paths_list[j]
            path2 = random.choice(img_paths2)
            pair_list.append([path1, path2])
            label_list.append(0)
            N_neg += 1
    for _ in range(N_pos - N_neg):
        ids = random.sample(idx_val, 2)
        img_paths1 = img_paths_list[ids[0]]
        img_paths2 = img_paths_list[ids[1]]
        path1 = random.choice(img_paths1)
        path2 = random.choice(img_paths2)
        pair_list.append([path1, path2])
        label_list.append(0)
        N_neg += 1
    # shuffle label and pair
    ids = list(range(len(label_list)))
    random.shuffle(ids)
    label_list = [label_list[i] for i in ids]
    pair_list = [pair_list[i] for i in ids]
    
    print(f'N_neg: {N_neg}')
    print(f'N_pos: {N_pos}')
    with open(f'{DATA_ROOT}/val.txt', 'w') as f:
        for pair, label in zip(pair_list, label_list):
            f.write(f'{pair[0]},{pair[1]},{label}\n')
    # save test.txt
    img_dirs = [f'{DATA_ROOT}/test_pair/{i}' for i in range(600)]
    img_paths_list = [[f'{dir}/A_a.jpg', f'{dir}/B_a.jpg'] for dir in img_dirs]
    with open(f'{DATA_ROOT}/test.txt', 'w') as f:
        for pair in img_paths_list:
            f.write(f'{pair[0]},{pair[1]}\n')


def correct_img(img_path):
    img = get_face_img(img_path, model='cnn', idx=1)
    img.save(img_path.replace('.jpg', '_a.jpg'))    



if __name__ == '__main__':
    crop_and_align_all_face()
    generate_data_set()
    correct_img(f'{DATA_ROOT}/test_pair/7/A.jpg')
    

            
                