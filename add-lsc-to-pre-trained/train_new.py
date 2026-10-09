import os
import tqdm
import glob
import torch
import random
import argparse
import torchvision
import pandas as pd
import bitsandbytes as bnb
import torch.nn.functional as F

from PIL import Image
from diffusers.optimization import get_scheduler
from diffusion_feature import FeatureExtractor, add_lsc
from safetensors.torch import save_file, safe_open, load_model
from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel


# ----- hyper parameters ----- #
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save_every_steps', type=int, default=200)
parser.add_argument('--start_from', type=str, default=None)
parser.add_argument('--resume_from', type=str, default=None)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--accumulation_step', type=int, default=4)
parser.add_argument('--input_root', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--lr_warmup_steps', type=int, default=500)
parser.add_argument('--prompt_dropout', type=int, default=0.1)
parser.add_argument('--reg_layer', type=str)
parser.add_argument('--reg_layer_end', type=str)
parser.add_argument('--reg_type', type=str, nargs='+')
parser.add_argument('--reg_weight', type=float, default=0.1)
parser.add_argument('--lsc_type', type=str, choices=['basic', 'res', 'eye'], default='basic')
parser.add_argument('--lsc_multiplier', type=float, default=1)
parser.add_argument('--force_learn', action='store_true')
parser.add_argument('--fl_weight', type=float, default=0.01)
args = parser.parse_args()

epochs = args.epochs
save_every_steps = args.save_every_steps
if args.start_from:
	start_from = args.start_from.replace('ga', "'ga'").replace('tv', "'tv'").replace('ss', "'ss'")
else:
	start_from = args.start_from
if args.resume_from:
	args.resume_from = args.resume_from.replace('ga', "'ga'").replace('tv', "'tv'").replace('ss', "'ss'")
lr = args.lr
weight_decay = args.weight_decay
accumulation_step = args.accumulation_step
lr_warmup_steps = args.lr_warmup_steps
prompt_dropout = args.prompt_dropout
reg_type = args.reg_type
reg_weight = args.reg_weight
lsc_type = 'basic'
lsc_multiplier = args.lsc_multiplier

input_root = args.input_root
output_dir = args.output_dir

output_dir = os.path.join(
	output_dir, 'scratch',
	f'lr{lr}-rw{reg_weight}-lm{lsc_multiplier}'
)

# ----- prepare models ----- #
if not args.resume_from:
	# here, we use from_config to randomly initialize a model and train it from scratch
	# rather than retrofit a pre-trained model
	config = PixArtTransformer2DModel.load_config('models/PixArt-Sigma', subfolder="transformer")
	transformer = PixArtTransformer2DModel.from_config(
		config, torch_dtype=torch.float32
	)
else:
	transformer = PixArtTransformer2DModel.from_pretrained(
		'models/PixArt-Sigma', subfolder='transformer', torch_dtype=torch.float32
	)

pipe = PixArtSigmaPipeline.from_pretrained(
	'models/PixArt-Sigma', transformer=transformer, torch_dtype=torch.float32
).to('cuda')
# pipe.transformer.to(torch.float32)
pipe.set_progress_bar_config(disable=True)
add_lsc(pipe.transformer, dim=1152, layer_count=len(pipe.transformer.transformer_blocks), lsc_type=lsc_type)
if args.resume_from:
	load_model(pipe.transformer.lsc, args.resume_from + 'lsc.safetensors')
pipe.unet = pipe.transformer

df = FeatureExtractor(
	layer={
		'pipe-gt': True, 'pipe-pred': True,# args.reg_layer: True, args.reg_layer_end: True,
		# 'lsc-15-before-res': True, 'lsc-15-after-res': True,
		# 'lsc-16-before-res': True, 'lsc-16-after-res': True,
		# 'lsc-17-before-res': True, 'lsc-17-after-res': True,
		# 'lsc-18-before-res': True, 'lsc-18-after-res': True,
		# 'lsc-19-before-res': True, 'lsc-19-after-res': True,
		# 'lsc-20-before-res': True, 'lsc-20-after-res': True,
		# 'lsc-21-before-res': True, 'lsc-21-after-res': True,
		# 'lsc-22-before-res': True, 'lsc-22-after-res': True,
		# 'lsc-23-before-res': True, 'lsc-23-after-res': True,
		# 'lsc-24-before-res': True, 'lsc-24-after-res': True,
		# 'lsc-25-before-res': True, 'lsc-25-after-res': True,
		# 'lsc-26-before-res': True, 'lsc-26-after-res': True,
		# 'lsc-27-before-res': True, 'lsc-27-after-res': True,
	},
	version='pixart-sigma',
	device='cuda',
	external_model=pipe,
	dtype='float32',
	train_unet=True,
)


# ----- prepare data ----- #
class DiffusionDataset(torch.utils.data.Dataset):
	def __init__(self, root_dir):
		super().__init__()
		self.image_root = os.path.join(root_dir, 'image')
		self.prompt_root = os.path.join(root_dir, 'prompt')

		self.all_images = sorted(glob.glob(os.path.join(self.image_root, '*')))
		self.all_prompts = sorted(glob.glob(os.path.join(self.prompt_root, '*')))

	def __len__(self):
		return len(self.all_images)

	def __getitem__(self, idx):
		image = self.all_images[idx]
		prompt = self.all_prompts[idx]

		image = Image.open(image)
		with open(prompt, 'r') as f:
			prompt = f.read()

		return image, prompt


# ----- training preparation ----- #
for p in pipe.transformer.parameters():
	p.requires_grad = True
all_params = set(pipe.transformer.parameters())
parameter_groups = [
	{'params': list(all_params), 'lr': lr},
]


optimizer = bnb.optim.AdamW8bit(parameter_groups, weight_decay=weight_decay)

dataset = DiffusionDataset(input_root)

lr_scheduler = get_scheduler(
	name="constant",
	optimizer=optimizer,
	num_warmup_steps=lr_warmup_steps * accumulation_step,
	num_training_steps=len(dataset) * accumulation_step * epochs,
)


# ----- training ----- #
def validate(df, step):
	os.makedirs(os.path.join(output_dir, f'{step}'), exist_ok=True)
	with torch.no_grad():
		df.enable_pipeline_generation()
		# with torch.autocast(dtype=torch.float16, device_type='cuda'):
		image = df.pipe(
			"a cat holding a paper with word prompt on it",
			image=Image.open(
				'datasets/SPair-71k/JPEGImages/cat/2007_005460.jpg').resize((1024, 1024)
			),
			height=1024,
			width=1024,
			strength=1,
		).images[0]
		image.save(os.path.join(output_dir, f'{step}', 'sample.png'))
		df.restart_feature_mode()

def gaussian_mse_loss(feat):
	denoised_map = torchvision.transforms.functional.gaussian_blur(
		feat, kernel_size=5
	)
	return F.mse_loss(feat, denoised_map, reduction="mean")

def total_variation_loss(img):
	bs_img, c_img, h_img, w_img = img.shape
	tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
	tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
	return (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

def self_similarity_loss(feat):
	a = feat[:,:,1:,1:]
	b = feat[:,:,:-1,:-1]
	return F.mse_loss(a, b, reduction='mean')

def whitening_loss_conv(x):
	"""
	Computes whitening loss for a 4D convolutional feature map (B, C, H, W).

	Args:
	    x (torch.Tensor): Input tensor of shape (B, C, H, W)

	Returns:
	    torch.Tensor: Scalar tensor representing the whitening loss.
	"""
	def whitening_loss(x):
		batch_size, num_features = x.shape
		x_centered = x - x.mean(dim=0)
		cov = (x_centered.T @ x_centered) / batch_size
		identity = torch.eye(num_features, device=x.device, dtype=x.dtype)
		loss = (cov - identity).pow(2).mean()
		return loss
	batch_size, channels, height, width = x.shape
	# Reshape to (B * H * W, C)
	x_reshaped = x.view(batch_size, channels, -1)         # (B, C, H*W)
	x_reshaped = x_reshaped.permute(0, 2, 1).contiguous() # (B, H*W, C)
	x_reshaped = x_reshaped.view(-1, channels)            # (B*H*W, C)

	return whitening_loss(x_reshaped)

step = 1
loss_to_print = 0.
for epoch in range(epochs):
	for image, prompt in tqdm.tqdm(dataset):
		if random.choices([True, False], weights=[prompt_dropout, 1 - prompt_dropout]):
			prompt = ''

		timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (1,), device='cuda')
		timesteps = timesteps.long()
		# with torch.autocast(dtype=torch.float16, device_type='cuda'):
		feat = df.extract(
			prompts=prompt,
			batch_size=1,
			image=[image],
			t=timesteps,
		)
		# print(feat['pipe-pred'])
		# print(feat['pipe-gt'])
		# print(pipe.transformer.lsc.all_lsc['20'].linear.weight)
		loss = F.mse_loss(feat['pipe-pred'].float(), feat['pipe-gt'].float(), reduction="mean")

		# if 'ga' in reg_type:
		# 	loss += reg_weight * gaussian_mse_loss(feat[args.reg_layer])
		# 	loss += reg_weight * gaussian_mse_loss(feat[args.reg_layer_end])
		# elif 'tv' in reg_type:
		# 	loss += reg_weight * total_variation_loss(feat[args.reg_layer])
		# 	loss += reg_weight * total_variation_loss(feat[args.reg_layer_end])
		# elif 'ss' in reg_type:
		# 	loss += reg_weight * self_similarity_loss(feat[args.reg_layer])
		# 	loss += reg_weight * self_similarity_loss(feat[args.reg_layer_end])
		# elif 'wh' in reg_type:
		# 	loss += reg_weight * whitening_loss_conv(feat[args.reg_layer])
		# 	loss += reg_weight * whitening_loss_conv(feat[args.reg_layer_end])
		# else:
		# 	raise NotImplementedError

		# if args.force_learn:
		# 	fl_loss = 0.
		# 	count = 0
		# 	for k, v in feat.items():
		# 		if 'before-res' not in k:
		# 			continue
		# 		before = v
		# 		after = feat[k.replace('before', 'after')]
		# 		fl_loss += - 0.01 * F.mse_loss(before, after, reduction='mean')
		# 		count += 1
		# 	fl_loss = fl_loss / count
		# 	loss += fl_loss * args.fl_weight

		loss.backward()
		if step % accumulation_step == 0:
			optimizer.step()
			optimizer.zero_grad()
			lr_scheduler.step()

		loss_to_print += loss.item()
		if step % 50 == 0:
			print(loss_to_print / 50)
			loss_to_print = 0.

		if step % save_every_steps == 0:
			validate(df, step)
			state_dict = pipe.transformer.lsc.state_dict()
			save_path = os.path.join(output_dir, f'{step}', 'lsc.safetensors')
			save_file(state_dict, save_path)
			tmp_lsc = pipe.transformer.lsc
			del pipe.transformer.lsc
			pipe.transformer.save_pretrained(os.path.join(output_dir, f'{step}', 'transformer'))
			pipe.transformer.lsc = tmp_lsc

		step += 1
