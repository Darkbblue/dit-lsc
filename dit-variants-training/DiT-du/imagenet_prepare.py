import os
import tqdm
import tarfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=int, required=True)
args = parser.parse_args()

root = '/jindofs_temp/users/292869/intern/benyuan/datasets/imagenet/root/'

file = f'{root}train_images_{args.i}.tar.gz'

with tarfile.open(file) as f:
    for img in tqdm.tqdm(f.getmembers()):
        name = img.name
        # print(name)
        class_name = name.split('_')[0]
        # print(class_name)
        # print(img)
        os.makedirs(f'{root}train/{class_name}', exist_ok=True)
        f.extract(img, path=f'{root}train/{class_name}/')
