################################################################################
# Copyright 2019-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# File utility functions
################################################################################

import os
import sys
sys.stdout.flush()
sys.path.insert(0, "../common")
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from hd5_explore import hdf_read

from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam, RMSprop, lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, Subset

from model import Model_3DCNN, strip_prefix_if_present
from data_reader import Dataset_MLHDF
from img_util import GaussianFilter, Voxelizer3D
from file_util import *


# set CUDA for PyTorch
use_cuda = torch.cuda.is_available()
cuda_count = torch.cuda.device_count()
if use_cuda:
	device = torch.device(args.device_name)
	torch.cuda.set_device(int(args.device_name.split(':')[1]))
else:
	device = torch.device("cpu")
print(use_cuda, cuda_count, device)


def worker_init_fn(worker_id):
	np.random.seed(int(0))

def train():

	# Read hd5 files for training and validation
	X_train,Y_train = hdf_read('postera_protease2_pos_neg_train.hdf5')
	train_data = TensorDataset(X_train,Y_train)

	X_val,Y_val = hdf_read('postera_protease2_pos_neg_val.hdf5')
	val_data = TensorDataset(X_val,Y_val)

	# check multi-gpus
	# check multi-gpus
	num_workers = 0
	if cuda_count > 1:
		num_workers = cuda_count

	batch_size = 32
	epoch_count = 50
	checkpoint_iter = 50
	# initialize data loader
	batch_count = len(Y_train) // batch_size
	dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=None)


	val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=None)

	# define voxelizer, gaussian_filter
	voxelizer = Voxelizer3D(use_cuda=use_cuda, verbose = 0)
	gaussian_filter = GaussianFilter(dim=3, channels=19, kernel_size=11, sigma=1, use_cuda=use_cuda)

	# define model
	model = Model_3DCNN(use_cuda=use_cuda, verbose = 1)
	if use_cuda:
		model = model.cuda()
	if cuda_count > 1:
		model = nn.DataParallel(model)

	model.to(device)
	
	if isinstance(model, (DistributedDataParallel, DataParallel)):
		model_to_save = model.module
	else:
		model_to_save = model

	# set loss, optimizer, decay, other parameters
	#if args.rmsd_weight == True:
	#	loss_fn = WeightedMSELoss().float()
	#else:
	loss_fn = nn.BCELoss().float()
	#optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
	optimizer = RMSprop(model.parameters(), lr = 0.0001)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

	# load model
	epoch_start = 0
	model_path = 'refined_model.pth'
	if valid_file(model_path):
		checkpoint = torch.load(model_path, map_location=device)
		model_state_dict = checkpoint.pop("model_state_dict")
		strip_prefix_if_present(model_state_dict, "module.")
		model_to_save.load_state_dict(model_state_dict, strict=False)
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		epoch_start = checkpoint["epoch"]
		loss = checkpoint["loss"]
		print("checkpoint loaded: %s" % model_path)

	#if not os.path.exists(os.path.dirname(model_path)):
	#	os.makedirs(os.path.dirname(model_path))
	#output_dir = os.path.dirname(model_path)

	step = 0
	for epoch_ind in range(epoch_start,epoch_count):
		vol_batch = torch.zeros((batch_size,19,48,48,48)).float().to(device)
		losses = []
		model.train()
		for batch_ind,batch in enumerate(dataloader):

			# transfer to GPU
			#if args.rmsd_weight == True:
			#	x_batch_cpu, y_batch_cpu, w_batch_cpu = batch
			#else:
			x_batch_cpu, y_batch_cpu = batch[0], batch[1]
			x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
			
			# voxelize into 3d volume
			for i in range(x_batch.shape[0]):
				xyz, feat = x_batch[i,:,:3], x_batch[i,:,3:]
				vol_batch[i,:,:,:,:] = voxelizer(xyz, feat)
			vol_batch = gaussian_filter(vol_batch)
			
			# forward training
			ypred_batch, _ = model(vol_batch[:x_batch.shape[0]])

			# compute loss
			#if args.rmsd_weight == True:
			#	loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float(), w_batch_cpu.float())
			#else:
			loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float())
				
			losses.append(loss.cpu().data.item())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			print("[%d/%d-%d/%d] training, loss: %.3f, lr: %.7f" % (epoch_ind+1, epoch_count, batch_ind+1, batch_count, loss.cpu().data.item(), optimizer.param_groups[0]['lr']))
			if step % checkpoint_iter == 0:
				checkpoint_dict = {
					"model_state_dict": model_to_save.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"loss": loss,
					"step": step,
					"epoch": epoch_ind
				}
				torch.save(checkpoint_dict, model_path)
				print("checkpoint saved: %s" % model_path)
			step += 1

		print("[%d/%d] training, epoch loss: %.3f" % (epoch_ind+1, epoch_count, np.mean(losses)))
		
		if val_dataset:
			val_losses = []
			model.eval()
			with torch.no_grad():
				for batch_ind, batch in enumerate(val_dataloader):

					x_batch_cpu, y_batch_cpu = batch
					x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
					
					for i in range(x_batch.shape[0]):
						xyz, feat = x_batch[i,:,:3], x_batch[i,:,3:]
						vol_batch[i,:,:,:,:] = voxelizer(xyz, feat)
					vol_batch = gaussian_filter(vol_batch)
					
					ypred_batch, _ = model(vol_batch[:x_batch.shape[0]])


					loss = loss_fn(ypred_batch.cpu().float(), y_batch_cpu.float())
						
					val_losses.append(loss.cpu().data.item())
					print("[%d/%d-%d/%d] validation, loss: %.3f" % (epoch_ind+1, epoch_count, batch_ind+1, batch_count, loss.cpu().data.item()))

				print("[%d/%d] validation, epoch loss: %.3f" % (epoch_ind+1, epoch_count, np.mean(val_losses)))

	# close dataset
	dataset.close()
	val_dataset.close()


def main():
	train()

if __name__ == "__main__":
	main()
