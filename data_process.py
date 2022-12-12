import cv2
import random
import numpy as np
import face_recognition as fr
from PIL import Image
from glob import glob
from tqdm import tqdm


def align_facee(face, landmarks):
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


def get_face_img(path):
    img = fr.load_image_file(path)
    landmarks = fr.face_landmarks(img)[0] # 68 points of face, dict
    aligned_face = align_facee(img, landmarks)
    face_locations = fr.face_locations(aligned_face)[0]
    aligned_face = crop_face(aligned_face, face_locations, 128)
    return aligned_face


DATA_ROOT = '/Users/liuyu/Downloads/Face'


if __name__ == '__main__':
    # process train data
    print('Processing train data...')
    img_dirs = sorted(glob(f'{DATA_ROOT}/train/*'))
    img_paths_list = [sorted(glob(f'{img_dir}/*.jpg')) for img_dir in img_dirs]
    for img_paths in tqdm(img_paths_list):
        for img_path in img_paths:
            aligned_face = get_face_img(img_path)
            aligned_face = Image.fromarray(aligned_face)
            # save img
            aligned_face.save(img_path.replace('.jpg', '_a.jpg'))
    # generate 300 val img_dirs
    N = 300
    random.seed(0)
    val_list = random.sample(img_dirs, N)
    train_list = [dir for dir in img_dirs if dir not in val_list]
    # save train.txt
    with open(f'{DATA_ROOT}/train.txt', 'w') as f:
        for dir in train_list:
            f.write(f'{dir}\n')
    # save val.txt
    with open(f'{DATA_ROOT}/val.txt', 'w') as f:
        for dir in val_list:
            f.write(f'{dir}\n')
    # process test data
    print('Processing test data...')
    img_dirs = sorted(glob(f'{DATA_ROOT}/test/*'))
    img_paths_list = [sorted(glob(f'{img_dir}/*.jpg')) for img_dir in img_dirs]
    for img_paths in tqdm(img_paths_list):
        for img_path in img_paths:
            aligned_face = get_face_img(img_path)
            aligned_face = Image.fromarray(aligned_face)
            # save img
            aligned_face.save(img_path.replace('.jpg', '_a.jpg'))
    # save test.txt
    with open(f'{DATA_ROOT}/test.txt', 'w') as f:
        for dir in img_dirs:
            f.write(f'{dir}\n')
    

    