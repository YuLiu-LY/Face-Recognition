import face_recognition as fr
from tqdm import tqdm
import numpy as np


DATA_ROOT = '/home/yuliu/Dataset/Face1'


def get_embedding(path):
    img = fr.load_image_file(path)
    H, W, _ = img.shape
    face_locations = [[0, W, H, 0]]
    face_encodings = fr.face_encodings(img, face_locations)
    return face_encodings[0]


def fr_acc():
    with open(f'{DATA_ROOT}/val.txt', 'r') as f:
        lines = f.read().splitlines()
        pairs = [line.split(',') for line in lines]
        img_files = [[pair[0], pair[1]] for pair in pairs]
        labels = [int(pair[2]) for pair in pairs]
    labels = np.array(labels)
    dists = []
    for img_paths in tqdm(img_files):
        path1 = img_paths[0]
        path2 = img_paths[1]
        emb1 = get_embedding(path1)
        emb2 = get_embedding(path2)
        dist = np.linalg.norm(emb1 - emb2)
        dists.append(dist)
    dists = np.stack(dists)
    thresholds = np.arange(0.1, 1, 0.01)
    accs = []
    for threshold in thresholds:
        pred = dists < threshold
        acc = np.mean(pred == labels) 
        accs.append(acc)
    accs = np.stack(accs)
    best_threshold = thresholds[np.argmax(accs)]
    print(f'Best threshold: {best_threshold}')
    print(f'Best accuracy: {np.max(accs)}')

if __name__ == '__main__':
    fr_acc()
