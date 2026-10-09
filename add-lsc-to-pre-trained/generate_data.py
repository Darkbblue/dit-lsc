import os
import tqdm
import torch
import argparse
import pandas as pd
from PIL import Image
from diffusers import PixArtSigmaPipeline, StableDiffusion3Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--slicing_count', type=int, default=1)
parser.add_argument('--slicing_idx', type=int, default=0)
args = parser.parse_args()


# pipe = PixArtSigmaPipeline.from_pretrained(
#     'models/PixArt-Sigma', torch_dtype=torch.float16
# ).to('cuda')
pipe = StableDiffusion3Pipeline.from_pretrained(
    'models/SD3.5M', torch_dtype=torch.bfloat16
).to('cuda')
pipe.set_progress_bar_config(disable=True)


metadata_table = pd.read_parquet(
    args.input_file
)
all_prompts = metadata_table['prompt']


os.makedirs(os.path.join(args.output_path, 'image'), exist_ok=True)
os.makedirs(os.path.join(args.output_path, 'prompt'), exist_ok=True)
image_placeholder = Image.open(
    'datasets/SPair-71k/JPEGImages/cat/2007_005460.jpg'
).resize((1024, 1024))
for i, prompt in enumerate(tqdm.tqdm(all_prompts)):
    if i % args.slicing_count != args.slicing_idx:
        continue
    prompt = prompt.rstrip()
    with torch.no_grad():
        image = pipe(
            prompt,
            # image=image_placeholder,
            height=1024,
            width=1024,
            # strength=1,
            guidance_scale=4.5,
        ).images[0]
    image.save(os.path.join(args.output_path, 'image', f'{i}.png'))

    with open(os.path.join(args.output_path, 'prompt', f'{i}.txt'), 'w') as f:
        f.write(prompt)
