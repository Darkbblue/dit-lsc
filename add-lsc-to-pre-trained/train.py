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
from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel, StableDiffusion3Img2ImgPipeline, SD3Transformer2DModel


# ----- hyper parameters ----- #
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2)
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
parser.add_argument('--reg_layer_list', type=str, nargs='+', default=[])
parser.add_argument('--reg_type', type=str, nargs='+')
parser.add_argument('--reg_weight', type=float, default=0.1)
parser.add_argument('--lsc_type', type=str, default='basic')
parser.add_argument('--lsc_multiplier', type=float, default=1)
parser.add_argument('--force_learn', action='store_true')
parser.add_argument('--fl_weight', type=float, default=0.01)
parser.add_argument('--force_down_noise', action='store_true')
parser.add_argument('--reg_layer_list_2', type=str, nargs='+', default=[])
parser.add_argument('--reg_weight_2', type=float, default=0.1)
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
lsc_type = args.lsc_type
lsc_multiplier = args.lsc_multiplier

input_root = args.input_root
output_dir = args.output_dir

train_backbone = (start_from is not None) or 'basic' not in lsc_type
if not train_backbone:
	output_dir = os.path.join(output_dir, 'stage1', f'lr{lr}-{lsc_type}-reg-{args.reg_layer}-{reg_type}-rw{reg_weight}-lm{lsc_multiplier}')
else:
	if 'basic' in lsc_type:
		output_dir = os.path.join(
			output_dir, 'stage2', start_from.split('/')[-2],
			f'lr{lr}-ac{accumulation_step}-reg-{args.reg_layer}-{reg_type}-rw{reg_weight}-lm{lsc_multiplier}'
		)
	else:
		if not args.force_learn:
			output_dir = os.path.join(
				output_dir, 'single-stage',
				f'{lsc_type}-reg-{args.reg_layer}-{reg_type}-rw{reg_weight}-lm{lsc_multiplier}-rf{args.resume_from}'
			)
		else:
			output_dir = os.path.join(
				output_dir, 'single-stage',
				f'fl-{lsc_type}-fw{args.fl_weight}-reg-{args.reg_layer}-{reg_type}-rw{reg_weight}-rf{args.resume_from}'
			)
		if args.force_down_noise:
			output_dir += '-fdn'
		if args.reg_weight_2:
			output_dir += f'-rw2{args.reg_weight_2}'


# ----- prepare models ----- #
if args.resume_from is None:
	# transformer = PixArtTransformer2DModel.from_pretrained(
	# 	'models/PixArt-Sigma', subfolder='transformer', torch_dtype=torch.float32
	# )
	transformer = SD3Transformer2DModel.from_pretrained(
		'models/SD3.5M', subfolder='transformer', torch_dtype=torch.float32
	)
else:
	# transformer = PixArtTransformer2DModel.from_pretrained(
	# 	args.resume_from + 'transformer', torch_dtype=torch.float32
	# )
	transformer = SD3Transformer2DModel.from_pretrained(
		args.resume_from + 'transformer', torch_dtype=torch.float32
	)
# pipe = PixArtSigmaPipeline.from_pretrained(
# 	'models/PixArt-Sigma', transformer=transformer, torch_dtype=torch.float32
# ).to('cuda')
pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
	'models/SD3.5M', transformer=transformer, torch_dtype=torch.float32
).to('cuda')
pipe.set_progress_bar_config(disable=True)
# skip_table = {
# 	'down': {1: 12, 3: 17, 5: 22},
# 	'up': {12: 1, 17: 3, 22: 5}
# }
# add_lsc(pipe.transformer, dim=1152, layer_count=len(pipe.transformer.transformer_blocks), lsc_type=lsc_type, skip_table=skip_table)
# add_lsc(pipe.transformer, dim=1152, layer_count=len(pipe.transformer.transformer_blocks), lsc_type=lsc_type)
add_lsc(pipe.transformer, dim=1536, layer_count=len(pipe.transformer.transformer_blocks), lsc_type=lsc_type)
if args.force_down_noise:
	pipe.transformer.lsc.skip_table['up'][21] = 2
	pipe.transformer.lsc.skip_table['up'][18] = 2
	pipe.transformer.lsc.skip_table['up'][15] = 4
print(pipe.transformer.lsc.skip_table)
if start_from:
	load_model(pipe.transformer.lsc, start_from)
if args.resume_from:
	load_model(pipe.transformer.lsc, args.resume_from + 'lsc.safetensors')
pipe.unet = pipe.transformer

if args.force_learn:
	layer = {
		'pipe-gt': True, 'pipe-pred': True,
		'lsc-15-before-res': True, 'lsc-15-after-res': True,
		'lsc-16-before-res': True, 'lsc-16-after-res': True,
		'lsc-17-before-res': True, 'lsc-17-after-res': True,
		'lsc-18-before-res': True, 'lsc-18-after-res': True,
		'lsc-19-before-res': True, 'lsc-19-after-res': True,
		'lsc-20-before-res': True, 'lsc-20-after-res': True,
		'lsc-21-before-res': True, 'lsc-21-after-res': True,
		'lsc-22-before-res': True, 'lsc-22-after-res': True,
		'lsc-23-before-res': True, 'lsc-23-after-res': True,
		'lsc-24-before-res': True, 'lsc-24-after-res': True,
		'lsc-25-before-res': True, 'lsc-25-after-res': True,
		'lsc-26-before-res': True, 'lsc-26-after-res': True,
		'lsc-27-before-res': True, 'lsc-27-after-res': True,
	}
else:
	layer = {
		'pipe-gt': True, 'pipe-pred': True,
	}

if args.reg_layer:
	layer[args.reg_layer] = True
for l in args.reg_layer_list:
	layer[l] = True
for l in args.reg_layer_list_2:
	layer[l] = True
# df = FeatureExtractor(
# 	layer=layer,
# 	version='pixart-sigma',
# 	device='cuda',
# 	external_model=pipe,
# 	dtype='float32',
# 	train_unet=True,
# )
df = FeatureExtractor(
	layer=layer,
	version='3.5-medium',
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
		# self.all_prompts = sorted(glob.glob(os.path.join(self.prompt_root, '*')))

	def __len__(self):
		return len(self.all_images)

	def __getitem__(self, idx):
		image = self.all_images[idx]
		# prompt = self.all_prompts[idx]
		prompt = image.replace('image', 'prompt').replace('png', 'txt')

		# print(image)
		image = Image.open(image)
		with open(prompt, 'r') as f:
			prompt = f.read()

		return image, prompt


# ----- training preparation ----- #
if train_backbone:
	for p in pipe.transformer.parameters():
		p.requires_grad = True
	all_params = set(pipe.transformer.parameters())
	lsc_params = set(pipe.transformer.lsc.parameters())
	backbone_params = all_params - lsc_params
	parameter_groups = [
		{'params': list(backbone_params), 'lr': lr},
		{'params': list(lsc_params), 'lr': lsc_multiplier * lr},
	]
else:
	for p in pipe.transformer.parameters():
		p.requires_grad = False
	for p in pipe.transformer.lsc.parameters():
		p.requires_grad = True
	parameters_to_train = list(filter(lambda p: p.requires_grad, pipe.transformer.parameters()))
	parameter_groups = [{"params": parameters_to_train, "lr": lr}]


optimizer = bnb.optim.AdamW8bit(parameter_groups, weight_decay=weight_decay)

dataset = DiffusionDataset(input_root)

lr_scheduler = get_scheduler(
	name="cosine",
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
		# image = df.pipe(
		# 	"a cat holding a paper with word prompt on it",
		# 	image=Image.open(
		# 		'datasets/SPair-71k/JPEGImages/cat/2007_005460.jpg').resize((1024, 1024)
		# 	),
		# 	height=1024,
		# 	width=1024,
		# 	strength=1,
		# ).images[0]
		image = df.pipe(
			prompt="a cat holding a paper with word prompt on it",
			image=Image.open(
				'datasets/SPair-71k/JPEGImages/cat/2007_005460.jpg').resize((1024, 1024)
			),
			height=1024,
			width=1024,
			guidance_scale=4.5,
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
		try:
			feat = df.extract(
				prompts=prompt,
				batch_size=1,
				image=[image],
				t=timesteps,
			)
		except RuntimeError as e:
			print('encountered and skipped:', e)
			continue
		# print(feat['pipe-pred'])
		# print(feat['pipe-gt'])
		# print(pipe.transformer.lsc.all_lsc['20'].linear.weight)
		loss = F.mse_loss(feat['pipe-pred'].float(), feat['pipe-gt'].float(), reduction="mean")

		if args.reg_layer:
			if 'ga' in reg_type:
				loss += reg_weight * gaussian_mse_loss(feat[args.reg_layer])
			elif 'tv' in reg_type:
				loss += reg_weight * total_variation_loss(feat[args.reg_layer])
			elif 'ss' in reg_type:
				loss += reg_weight * self_similarity_loss(feat[args.reg_layer])
			elif 'wh' in reg_type:
				loss += reg_weight * whitening_loss_conv(feat[args.reg_layer])
			else:
				raise NotImplementedError
		for l in args.reg_layer_list:
			if 'ga' in reg_type:
				loss += reg_weight * gaussian_mse_loss(feat[l]) / (len(args.reg_layer_list) + len(args.reg_layer_list_2))
			elif 'tv' in reg_type:
				loss += reg_weight * total_variation_loss(feat[l]) / (len(args.reg_layer_list) + len(args.reg_layer_list_2))
			elif 'ss' in reg_type:
				loss += reg_weight * self_similarity_loss(feat[l]) / (len(args.reg_layer_list) + len(args.reg_layer_list_2))
			elif 'wh' in reg_type:
				loss += reg_weight * whitening_loss_conv(feat[l]) / (len(args.reg_layer_list) + len(args.reg_layer_list_2))
			else:
				raise NotImplementedError
		for l in args.reg_layer_list_2:
			if 'ga' in reg_type:
				loss += args.reg_weight_2 * gaussian_mse_loss(feat[l]) / (len(args.reg_layer_list) + len(args.reg_layer_list_2))
			elif 'tv' in reg_type:
				loss += args.reg_weight_2 * total_variation_loss(feat[l]) / (len(args.reg_layer_list) + len(args.reg_layer_list_2))
			elif 'ss' in reg_type:
				loss += args.reg_weight_2 * self_similarity_loss(feat[l]) / (len(args.reg_layer_list) + len(args.reg_layer_list_2))
			elif 'wh' in reg_type:
				loss += args.reg_weight_2 * whitening_loss_conv(feat[l]) / (len(args.reg_layer_list) + len(args.reg_layer_list_2))
			else:
				raise NotImplementedError

		if args.force_learn:
			fl_loss = 0.
			count = 0
			for k, v in feat.items():
				if 'before-res' not in k:
					continue
				before = v
				after = feat[k.replace('before', 'after')]
				fl_loss += - 0.01 * F.mse_loss(before, after, reduction='mean')
				count += 1
			fl_loss = fl_loss / count
			loss += fl_loss * args.fl_weight

		if reg_weight < 0 or args.reg_weight_2 < 0:
			torch.nn.utils.clip_grad_norm_(pipe.transformer.parameters(), 1.0, norm_type=2.0)

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
			if train_backbone:
				tmp_lsc = pipe.transformer.lsc
				del pipe.transformer.lsc
				pipe.transformer.save_pretrained(os.path.join(output_dir, f'{step}', 'transformer'))
				pipe.transformer.lsc = tmp_lsc

		step += 1
