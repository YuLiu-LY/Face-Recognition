import os
import cv2
import random
import numpy as np
import face_recognition as fr
from PIL import Image
from glob import glob
from tqdm import tqdm


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
    # crop face
    return rotated_face

def crop_face(face, location, img_size):
    '''
    Crop face by face location
    face: (H, W, C)
    location: (top, right, bottom, left)
    '''
    face = face[location[0]:location[2], location[3]:location[1]]
    face = cv2.resize(face, (img_size, img_size))
    return face


def get_face_img(path, model='hog'):
    img = fr.load_image_file(path)
    location = fr.face_locations(img, model=model)
    if len(location) == 0:
        return None
    landmarks = fr.face_landmarks(img, location)[0] # 68 points of face, dict
    aligned_face = align_face(img, landmarks)
    location = fr.face_locations(aligned_face, model=model)
    if len(location) == 0:
        return None
    aligned_face = crop_face(aligned_face, location[0], 128)
    return aligned_face


DATA_ROOT = '/home/yuliu/Dataset/Face'

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
            aligned_face = Image.fromarray(aligned_face)
            # save img
            aligned_face.save(img_path.replace('.jpg', '_a.jpg'))
    # process failed data with cnn model
    print(f'Processing failed data: {len(failed_list)}...')
    for img_path in tqdm(failed_list):
        aligned_face = get_face_img(img_path, model='cnn')
        if aligned_face is None:
            continue
        aligned_face = Image.fromarray(aligned_face)
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
            aligned_face = Image.fromarray(aligned_face)
            # save img
            aligned_face.save(img_path.replace('.jpg', '_a.jpg'))

def generate_data_set():
    # generate validation set
    img_dirs = sorted(glob(f'{DATA_ROOT}/training_set/*'))
    img_paths_list = [sorted(glob(f'{img_dir}/*_a.jpg')) for img_dir in img_dirs]
    N = 300
    random.seed(0)
    idx = random.sample(range(len(img_dirs)), N)
    train_dirs = [img_dirs[i] for i in range(len(img_dirs)) if i not in idx]
    # save train.txt
    with open(f'{DATA_ROOT}/train.txt', 'w') as f:
        for dir in train_dirs:
            f.write(f'{dir}\n')
    # save val.txt
    # generate pairs of same face and different face
    pair_list = []
    label_list = []
    N_pos = 0
    for i in idx:
        img_paths = img_paths_list[i]
        if len(img_paths) == 0:
            print(f'No face in {img_dirs[i]}')
            continue
        if len(img_paths) >= 2:
            pair_list.append(random.sample(img_paths, 2))
            label_list.append(1)
            N_pos += 1
        else:
            path1 = img_paths[0]
            img_paths2 = img_paths_list[random.choice(idx)]
            path2 = random.choice(img_paths2)
            pair_list.append([path1, path2])
            label_list.append(0)
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
    

if __name__ == '__main__':
    # crop_and_align_all_face()
    generate_data_set()