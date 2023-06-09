# Core libraries
import os
import sys
import argparse
from tqdm import tqdm

#numpy
import numpy as np

# PyTorch stuff
import torch
from torch.autograd import Variable

# Local libraries
from utilities.loss import *
from utilities.mining_utils import *
from utilities.utils import Utilities


#Load DNA embeddings
#------------------------------------------------------------------------------
DNA_embeddings = np.load('/content/drive/MyDrive/sgt/DNA_embeddings/DNA_embeddings_avarage.npy')
DNA_embeddings_pos_neg = np.load('/content/drive/MyDrive/sgt/DNA_embeddings/DNA_embeddings_sorted.npy')
DNA_embeddings_pos_neg_labels = np.load('/content/drive/MyDrive/sgt/DNA_embeddings/DNA_labels_sorted.npy')

#convert to tensor
DNA_embeddings_torch = torch.from_numpy(DNA_embeddings)
DNA_embeddings_pos_neg_torch = torch.from_numpy(DNA_embeddings_pos_neg)

#load to cuda
DNA_embeddings_torch = Variable(DNA_embeddings_torch.cuda())
DNA_embeddings_pos_neg_torch = Variable(DNA_embeddings_pos_neg_torch.cuda())
#---------TBC @ #Get the embeddings/predictions for each-----------------------



"""
File is for training the network via cross fold validation
"""

# Let's cross validate
def crossValidate(args):
	# Loop through each fold for cross validation
	for k in range(args.fold_number, args.num_folds):
		print(f"Beginning training for fold {k+1} of {args.num_folds}")

		# Directory for storing data to do with this fold
		args.fold_out_path = os.path.join(args.out_path, f"fold_{k}")

		# Create a folder in the results folder for this fold as well as to store embeddings
		os.makedirs(args.fold_out_path, exist_ok=True)

		# Store the current fold
		args.current_fold = k

		# Let's train!
		trainFold(args)

# Train for a single fold
def trainFold(args):
	# Create a new instance of the utilities class for this fold
	utils = Utilities(args)

	# Let's prepare the objects we need for training based on command line arguments
	data_loader, model, model_2, loss_fn, loss_fn_2, optimiser, optimiser_2 = utils.setupForTraining(args)

	# Training tracking variables
	global_step = 0 
	accuracy_best = 0
	epoch_number = 1

	# Main training loop
	for epoch in tqdm(range(args.num_epochs), desc="Training epochs"):
		'''
        #----------------------fcl
		if epoch_number > 0:
			for images, images_pos, images_neg, labels, labels_neg in data_loader:
                # Put the images on the GPU and express them as PyTorch variables
				images = Variable(images.cuda())
				images_pos = Variable(images_pos.cuda())
				images_neg = Variable(images_neg.cuda())
                
				if "Softmax" in args.loss_function:
					embed_anch_im, embed_pos, embed_neg, preds = model(images, images_pos, images_neg)
    				#DNA embeddings -----------------------------------------------
					labels_tensor_1d = torch.reshape(labels, (-1,))
					label_indeces = labels_tensor_1d-1
					embed_anch_DNA_ = DNA_embeddings_torch[label_indeces]
					embed_anch_DNA = model_2(embed_anch_DNA_.float())
    				#print('labels',labels)
    				#print('labels_neg',labels_neg)
    				#print('label_indices',label_indeces)
    				#print('emb_anc_dna_sh', embed_anch.shape)
    				#print('emb_anc_dna', embed_anch)
					labels_neg_tensor_1d_ = torch.reshape(labels_neg, (-1,))
					label_indeces_neg = labels_neg_tensor_1d_-1
					embed_neg_DNA_ = DNA_embeddings_torch[label_indeces_neg] 
					embed_neg_DNA = model_2(embed_neg_DNA_.float())
                    
                 # Backprop and optimise_FCL
				optimiser_2.zero_grad()
				loss_2, triplet_loss_2, loss_softmax_2 = loss_fn_2(embed_pos, embed_anch_DNA, embed_neg_DNA, 0, labels, labels_neg)
     				#loss_2, triplet_loss_2, loss_softmax_2 = loss_fn_2(embed_anch_DNA, embed_anch_DNA, embed_neg_DNA, preds.detach(),labels, labels_neg)
				triplet_loss_2.backward()#retain_graph=True)
				optimiser_2.step()
     				#torch.autograd.set_detect_anomaly(True)
                     
				global_step += 1
    
    
         			# Log the loss if its time to do so
				if global_step % args.logs_freq == 0:
					if "Softmax" in args.loss_function:
						utils.logTrainInfo(	epoch, global_step, loss_2.item(), 
                    	loss_triplet=triplet_loss_2.item(),
         				loss_triplet_2=triplet_loss_2.item(),
         				loss_softmax=0)
					else:
	 					utils.logTrainInfo(epoch, global_step, loss_2.item()) 

        #----------------fcl_end
        '''
        
        #----------------REsnet Train
		if epoch_number > 0:
			#utils.saveCheckpoint(epoch, model, optimiser, "current")
		# Mini-batch training loop over the training set
			for images, images_pos, images_neg, labels, labels_neg in data_loader:
			# Put the images on the GPU and express them as PyTorch variables
				images = Variable(images.cuda())
				images_pos = Variable(images_pos.cuda())
				images_neg = Variable(images_neg.cuda())
		   
			# Zero the optimiser
			#optimiser.zero_grad()
			#optimiser_2.zero_grad()
    
			# Get the embeddings/predictions for each
				if "Softmax" in args.loss_function:
					embed_anch_im, embed_pos, embed_neg, preds = model(images, images_pos, images_neg)
				#DNA embeddings -----------------------------------------------
					labels_tensor_1d = torch.reshape(labels, (-1,))
					label_indeces = labels_tensor_1d-1
					embed_anch_DNA_ = DNA_embeddings_torch[label_indeces]
					embed_anch_DNA = embed_anch_DNA_.float()
				#print('labels',labels)
				#print('labels_neg',labels_neg)
				#print('label_indices',label_indeces)
				#print('emb_anc_dna_sh', embed_anch.shape)
				#print('emb_anc_dna', embed_anch)
					labels_neg_tensor_1d_ = torch.reshape(labels_neg, (-1,))
					label_indeces_neg = labels_neg_tensor_1d_-1
					embed_neg_DNA_ = DNA_embeddings_torch[label_indeces_neg] 
					embed_neg_DNA = embed_neg_DNA_.float()
				#print('shape_emb_anc_model', embed_anch_DNA.shape) ### DNAEmbeddings----
				#print('shape_emb_anc_', embed_anch_DNA_.shape) ### DNAEmbeddings----
				#print('_emb_anc_', embed_anch_DNA_) ### DNAEmbeddings----
				#print('_emb_anc_model', embed_anch_DNA) ### DNAEmbeddings----



				#--------------------------------------------------------------


				else:
					embed_anch_im, embed_pos, embed_neg = model(images, images_pos, images_neg)
				#print('labels', labels)
				#print('type_labels', type(labels))
				#print('shape_labels', labels.shape)
				#print('emb_anc_im', embed_anch_im)
				#print('type_emb_anc', type(embed_anch))
				#print('shape_emb_anc', embed_anch.shape) ### DNAEmbeddings----
					labels_tensor_1d = torch.reshape(labels, (-1,))
					label_indeces = labels_tensor_1d-1
					embed_anch_DNA_ = DNA_embeddings_torch[label_indeces]
					embed_anch_DNA = embed_anch_DNA_.float()
				#print('shape_emb_anc_model', embed_anch_DNA.shape) ### DNAEmbeddings----
				#print('shape_emb_anc_', embed_anch_DNA_.shape) ### DNAEmbeddings----
				#print('_emb_anc_', embed_anch_DNA_.shape) ### DNAEmbeddings----
				#print('_emb_anc_model', embed_anch_DNA.shape) ### DNAEmbeddings----




        
					labels_neg_tensor_1d_ = torch.reshape(labels_neg, (-1,))
					label_indeces_neg = labels_neg_tensor_1d_-1
					embed_neg_DNA_ = DNA_embeddings_torch[label_indeces_neg]
					embed_neg_DNA = embed_neg_DNA_.float()
				#--------------------------------------------------------------
				#print('emb_anc_dna', embed_anch.shape)



			# Calculate the loss on this minibatch
				if "Softmax" in args.loss_function:
					#loss, triplet_loss, loss_softmax = loss_fn(embed_anch_DNA, embed_pos, embed_neg, preds, labels, labels_neg)
					loss, triplet_loss, loss_softmax = loss_fn(embed_anch_DNA, embed_anch_DNA, embed_neg_DNA, embed_anch_im, embed_pos, embed_neg, preds, labels, labels_neg)

				#loss_2, triplet_loss_2, loss_softmax_2 = loss_fn_2(embed_pos, embed_anch_DNA, embed_neg_DNA, preds, labels, labels_neg)

				else:
					loss = loss_fn(embed_anch_DNA, embed_pos.detach(), embed_neg.detach(), labels, labels_neg)
					#loss_2 = loss_fn_2(embed_pos, embed_anch_DNA, embed_neg_DNA, labels)
    
    
    # Backprop and optimise_FCL
			#if global_step%(2*epoch_number) == 1:
			#	optimiser_2.zero_grad()
			#	loss_2, triplet_loss_2, loss_softmax_2 = loss_fn_2(embed_pos, embed_anch_DNA.detach(), embed_neg_DNA.detach(), 0, labels, labels_neg)
				#loss_2, triplet_loss_2, loss_softmax_2 = loss_fn_2(embed_anch_DNA, embed_anch_DNA, embed_neg_DNA, preds.detach(),labels, labels_neg)
			#	triplet_loss_2.backward()#retain_graph=True)
			#	optimiser_2.step()
				#torch.autograd.set_detect_anomaly(True)
    
			# Backprop and optimise resnet
				optimiser.zero_grad()
				loss.backward()#retain_graph=True)
				optimiser.step()

			#global_step += 1

    
			#for param in model_2.parameters():
			#	print(param.data)
    
				global_step += 1



			# Log the loss if its time to do so
				if global_step % args.logs_freq == 0:
					if "Softmax" in args.loss_function:
						utils.logTrainInfo(	epoch, global_step, loss.item(), 
										loss_triplet=triplet_loss.item(), 
										loss_triplet_2=triplet_loss.item(),
										loss_softmax=loss_softmax)
					else:
						utils.logTrainInfo(epoch, global_step, loss.item()) 
		
        
        #----------------REsnet Train end

        
		epoch_number+=1



		# Every x epochs, let's evaluate on the validation set
		if epoch % args.eval_freq == 0:
			# Temporarily save model weights for the evaluation to use
			utils.saveCheckpoint(epoch, model, optimiser, "current")
			#utils.saveCheckpoint(epoch, model_2, optimiser, "current_fcl")


			# Test on the validation set
			accuracy_curr = utils.test(global_step)

			# Save the model weights as the best if it surpasses the previous best results
			if accuracy_curr > accuracy_best:
				utils.saveCheckpoint(epoch, model, optimiser, "best")
				#utils.saveCheckpoint(epoch, model_2, optimiser, "best_fcl")

				accuracy_best = accuracy_curr

# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Parameters for network training')

	# File configuration (the only required arguments)
	parser.add_argument('--out_path', type=str, default="", required=True,
						help="Path to folder to store results in")
	parser.add_argument('--folds_file', type=str, default="", required=True,
						help="Path to json file containing folds")

	# Core settings
	parser.add_argument('--num_folds', type=int, default=1,
						help="Number of folds to cross validate across")
	parser.add_argument('--fold_number', type=int, default=0,
						help="The fold number to START at")
	parser.add_argument('--dataset', type=str, default='OpenSetCows2020',
						help='Which dataset to use')
	parser.add_argument('--model', type=str, default='TripletResnetSoftmax',
						help='Which model to use: [TripletResnetSoftmax, TripletResnet]')
	parser.add_argument('--triplet_selection', type=str, default='HardestNegative',
						help='Which triplet selection method to use: [HardestNegative, RandomNegative,\
						SemihardNegative, AllTriplets]')
	parser.add_argument('--loss_function', type=str, default='OnlineReciprocalSoftmaxLoss',
						help='Which loss function to use: [TripletLoss, TripletSoftmaxLoss, \
						OnlineTripletLoss, OnlineTripletSoftmaxLoss, OnlineReciprocalTripletLoss, \
						OnlineReciprocalSoftmaxLoss]')

	# Hyperparameters
	parser.add_argument('--img_rows', nargs='?', type=int, default=224, 
						help='Height of the input image')
	parser.add_argument('--img_cols', nargs='?', type=int, default=224, 
						help='Height of the input image')
	parser.add_argument('--embedding_size', nargs='?', type=int, default=128, 
						help='dense layer size for inference')
	parser.add_argument('--num_epochs', nargs='?', type=int, default=500, 
						help='# of the epochs to train for')
	parser.add_argument('--batch_size', nargs='?', type=int, default=16,
						help='Batch Size')
	parser.add_argument('--learning_rate', type=float, default=0.001,
						help="Optimiser learning rate")
	parser.add_argument('--weight_decay', type=float, default=1e-4,
						help="Weight decay")
	parser.add_argument('--triplet_margin', type=float, default=0.5,
						help="Margin parameter for triplet loss")

	# Training settings
	parser.add_argument('--eval_freq', nargs='?', type=int, default=2,
						help='Frequency for evaluating model [epochs num]')
	parser.add_argument('--logs_freq', nargs='?', type=int, default=200,
						help='Frequency for saving logs [steps num]')

	args = parser.parse_args()

	# Let's cross validate!
	crossValidate(args)
