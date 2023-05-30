# Core libraries
import numpy as np
import math


# PyTorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
File contains loss functions selectable during training
"""

'''
class TripletLoss(nn.Module):
	def __init__(self, margin=4.0):
		super(TripletLoss, self).__init__()
		self.margin = margin
					
	def forward(self, anchor, positive, negative, labels):
		distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
		distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
		losses = F.relu(distance_positive - distance_negative + self.margin)
	
		return losses.sum()
'''

class TripletSoftmaxLoss(nn.Module):
	def __init__(self, margin=0.0, lambda_factor=0.01):
		super(TripletSoftmaxLoss, self).__init__()
		self.margin = margin
		self.loss_fn = nn.CrossEntropyLoss()
		self.lambda_factor = lambda_factor
					
	def forward(self, anchor, positive, negative, outputs, labels ):
		#distance_positive = torch.abs(anchor - positive).sum(1)
		#distance_negative = torch.abs(anchor - negative).sum(1)
		#losses = F.relu(distance_positive - distance_negative + self.margin)
		loss_softmax = self.loss_fn(input=outputs, target=labels)
		loss_total = loss_softmax # self.lambda_factor*losses.sum() + loss_softmax

		return loss_softmax,loss_softmax,loss_softmax# loss_total, losses.sum(), loss_softmax

class OnlineTripletLoss(nn.Module):
	def __init__(self, triplet_selector, margin=0.0):
		super(OnlineTripletLoss, self).__init__()
		self.margin = margin
		self.triplet_selector = triplet_selector

	def forward(self, anchor_embed, pos_embed, neg_embed, labels):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0)

		# Get the (e.g. hardest) triplets in this minibatch
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, labels)
		#print('num triplets', num_triplets)

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		if num_triplets == 0:
			ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
		else:
			# Use CUDA if we can
			if anchor_embed.is_cuda: triplets = triplets.cuda()

			# Compute triplet loss over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
            

		# Compute the losses
		losses = F.relu(ap_distances - an_distances + self.margin)

		return losses.mean()

class OnlineTripletSoftmaxLoss(nn.Module):
	def __init__(self, triplet_selector, margin=0.0, lambda_factor=0.01):
		super(OnlineTripletSoftmaxLoss, self).__init__()
		self.margin = margin
		self.loss_fn = nn.CrossEntropyLoss()
		self.lambda_factor = lambda_factor
		self.triplet_selector = triplet_selector
					
	def forward(self, anchor_embed, pos_embed, neg_embed, preds, labels, labels_neg):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0)

		# Define the labels as variables and put on the GPU
		gpu_labels = labels.view(len(labels))
		gpu_labels_neg = labels_neg.view(len(labels_neg))
		gpu_labels = Variable(gpu_labels.cuda())
		gpu_labels_neg = Variable(gpu_labels_neg.cuda())

		# Concatenate labels for softmax/crossentropy targets
		target = torch.cat((gpu_labels, gpu_labels, gpu_labels_neg), dim=0)

		# Get the (e.g. hardest) triplets in this minibatch
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, labels)

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		if num_triplets == 0:
			ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
		else:
			# Use CUDA if we can
			if anchor_embed.is_cuda: triplets = triplets.cuda()

			# Compute triplet loss over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
		
		# Compute the triplet losses
		triplet_losses = F.relu(ap_distances - an_distances + self.margin)

		# Compute softmax loss		
		loss_softmax = self.loss_fn(input=preds, target=target-1)

		# Compute the total loss
		loss_total = triplet_losses.mean() #self.lambda_factor*triplet_losses.mean() #+ loss_softmax

		# Return them all!
		return loss_total, triplet_losses.mean(), loss_softmax

# Reciprocal triplet loss from 
# "Who Goes There? Exploiting Silhouettes and Wearable Signals for Subject Identification
# in Multi-Person Environments"
'''
class OnlineReciprocalTripletLoss(nn.Module):
	def __init__(self, triplet_selector):
		super(OnlineReciprocalTripletLoss, self).__init__()
		self.triplet_selector = triplet_selector

	def forward(self, anchor_embed, pos_embed, neg_embed, labels, labels_neg):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0)
		target = torch.cat((labels, labels, labels_neg)).cuda()
        

		# Get the (e.g. hardest) triplets in this minibatch
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, target)

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		if num_triplets == 0:
			ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
		else:
			# Use CUDA if we can
			if anchor_embed.is_cuda: triplets = triplets.cuda()

			# Compute distances over the selected triplets
			#ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			#an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)

		# Actually compute reciprocal triplet loss
		losses = ap_distances + (1/an_distances)

		return losses.mean()
'''
#loss RESNET
class OnlineReciprocalTripletLoss(nn.Module): # 2 modalities
	def __init__(self, triplet_selector, margin=0.0,  lambda_factor=0.01):
		super(OnlineReciprocalTripletLoss, self).__init__()
		self.margin = margin
		self.loss_fn = nn.CrossEntropyLoss()
		self.lambda_factor = lambda_factor
		self.triplet_selector = triplet_selector
					
	def forward(self, anchor_embed_DNA, pos_embed_DNA, neg_embed_DNA, anchor_embed_vis, pos_embed_vis, neg_embed_vis, preds, labels, labels_neg):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed_DNA, pos_embed_vis, neg_embed_vis), dim=0) #works
		embeddings2 = torch.cat((anchor_embed_vis, pos_embed_vis, neg_embed_vis), dim=0) #works

        
        

		# Define the labels as variables and put on the GPU
		gpu_labels = labels.view(len(labels))
		gpu_labels_neg = labels_neg.view(len(labels_neg))
		gpu_labels = Variable(gpu_labels.cuda())
		gpu_labels_neg = Variable(gpu_labels_neg.cuda())

		# Concatenate labels for softmax/crossentropy targets
		target_softmax = torch.cat((gpu_labels, gpu_labels, gpu_labels_neg), dim=0)

		target = torch.cat((labels, labels, labels_neg)).cuda()


		# Get the (e.g. hardest) triplets in this minibatch #Tripletselection!!
		#triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, target)  #embeding_size=48 - how does it work??? labels should be target
        

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		num_triplets = 0
		if num_triplets == 0:
			#ap_distances = (anchor_embed_DNA - pos_embed_vis).pow(2).sum(1)
			#ap_distances2 = (anchor_embed_vis - pos_embed_vis).pow(2).sum(1)
			#an_distances = (anchor_embed_DNA - neg_embed_vis).pow(2).sum(1) 
			#an_distances2 =(anchor_embed_vis - neg_embed_vis).pow(2).sum(1)

			cos = nn.CosineSimilarity(dim=1, eps=1e-6)
			ap_distances = 1 - cos(anchor_embed_DNA, pos_embed_vis)
			term2 = F.relu(cos(anchor_embed_DNA, neg_embed_vis) - 0.5)

			#an_distances = 1 - cos(anchor_embed_DNA, neg_embed_vis)

		'''
		else:
			# Use CUDA if we can
			if anchor_embed.is_cuda: triplets = triplets.cuda()

			# Compute triplet loss over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)			
		'''

		# Compute the triplet losses
		#triplet_losses = ap_distances + 2*ap_distances2  + (2/(an_distances2)) # + (1/(an_distances))
		#triplet_losses = ap_distances + (1/(10*an_distances + 2))
		triplet_losses = ap_distances + term2
        

        
		#if triplet_losses.mean()>100 or math.isnan(triplet_losses.mean()):
			#print('resnet_ap = ',ap_distances)
			#print('resnet_ap2 = ',ap_distances)
			#print('rsenet_an = ', an_distances)
			#print('resnet_an2 = ', an_distances)
			#print('res_anchor_embed_DNA',anchor_embed_DNA)
			#print('res_anchor_embed_DNA',anchor_embed_DNA)
			#print('res_anchor_embed_vis',anchor_embed_vis)
			#print('res_pos_embed_vis',pos_embed_vis)

            
            
            
		#triplet_losses = ap_distances/mu + (1/(an_distances/mu))

		# Compute softmax loss		
		#loss_softmax = self.loss_fn(input=preds, target=target_softmax-1)
		loss_softmax = 1
		# Compute the total loss
		#loss_total = self.lambda_factor*triplet_losses.mean() + loss_softmax
		loss_total = triplet_losses.mean() #+ loss_softmax

		# Return them all!
		return loss_total, triplet_losses.mean(), loss_softmax

#to do loss fcn
class TripletLoss(nn.Module):
	def __init__(self, triplet_selector, margin=0.0,  lambda_factor=0.01):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.loss_fn = nn.CrossEntropyLoss()
		self.lambda_factor = lambda_factor
		self.triplet_selector = triplet_selector
					
	def forward(self, anchor_embed_DNA, pos_embed_DNA, neg_embed_DNA, anchor_embed_vis, pos_embed_vis, neg_embed_vis, preds, labels, labels_neg):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed_DNA, pos_embed_vis, neg_embed_vis), dim=0) #works
		embeddings2 = torch.cat((anchor_embed_vis, pos_embed_vis, neg_embed_vis), dim=0) #works

		# Define the labels as variables and put on the GPU
		gpu_labels = labels.view(len(labels))
		gpu_labels_neg = labels_neg.view(len(labels_neg))
		gpu_labels = Variable(gpu_labels.cuda())
		gpu_labels_neg = Variable(gpu_labels_neg.cuda())

		# Concatenate labels for softmax/crossentropy targets
		target_softmax = torch.cat((gpu_labels, gpu_labels, gpu_labels_neg), dim=0)

		target = torch.cat((labels, labels, labels_neg)).cuda()

		# Get the (e.g. hardest) triplets in this minibatch #Tripletselection!!
		#triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, target)  #embeding_size=48 - how does it work??? labels should be target
        
		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		num_triplets = 0
		if num_triplets == 0: #faulty !!
			ap_distances = (anchor_embed_DNA - pos_embed_DNA).pow(2).sum(1)
			print('ap1 = ',ap_distances)
			ap_distances2 = (anchor_embed_vis - anchor_embed_DNA).pow(2).sum(1)
			print('ap2 = ', ap_distances2)
			
			Recip_an_distances = (1/((anchor_embed_DNA - neg_embed_DNA).pow(2).sum(1)))
			print('an1 = ', Recip_an_distances)

			Recip_an_distances2 = (1/((anchor_embed_vis - anchor_embed_vis).pow(2).sum(1)))
			print('an2 = ',Recip_an_distances)

			
		else:
			# Use CUDA if we can
			if anchor_embed.is_cuda: triplets = triplets.cuda()

			# Compute triplet loss over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)			
		
		# Compute the triplet losses
		triplet_losses = ap_distances + ap_distances2 + Recip_an_distances + Recip_an_distances2#(1/(an_distances))
		print('triplet_losses = ',triplet_losses)

		#triplet_losses = ap_distances/mu + (1/(an_distances/mu))

		# Compute softmax loss		
		#loss_softmax = self.loss_fn(input=preds, target=target_softmax-1)
		loss_softmax = 1
		# Compute the total loss
		#loss_total = self.lambda_factor*triplet_losses.mean() + loss_softmax
		loss_total = triplet_losses.mean() #+ loss_softmax

		# Return them all!
		return loss_total, triplet_losses.mean(), loss_softmax



# Reciprocal triplet loss from - currently for fcn
# "Who Goes There? Exploiting Silhouettes and Wearable Signals for Subject Identification
# in Multi-Person Environments"
class OnlineReciprocalSoftmaxLoss(nn.Module):
	def __init__(self, triplet_selector, margin=0.0,  lambda_factor=0.01):
		super(OnlineReciprocalSoftmaxLoss, self).__init__()
		self.margin = margin
		self.loss_fn = nn.CrossEntropyLoss()
		self.lambda_factor = lambda_factor
		self.triplet_selector = triplet_selector
					
	def forward(self, anchor_embed, pos_embed, neg_embed, preds, labels, labels_neg):
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0) #works
        #save
		#torch.save(embeddings, 'embeddings.pt')
		#torch.save(labels, 'labels.pt')
		#torch.save(labels_neg, 'labels_neg.pt')

        
		#print('a=',anchor_embed)
		#print('a_sh=',anchor_embed.shape)
		#print('p=',pos_embed)
		#print('p_sh=',pos_embed.shape)
		#print('n=',neg_embed)
		#print('n_sh=',neg_embed)
		#print('emb', embeddings)
		#print('emb_sh',embeddings.shape)
		#print('emb[0]=', embeddings[0])
		#print('emb[16]=', embeddings[16])
		#print('emb[32]=', embeddings[32])



		#print('emb_p=',embeddings[:, 1])
		#print('emb_n=',embeddings[:, 2])
        
		#print('labels shape', labels.shape)
		#print(type(labels))

		# Define the labels as variables and put on the GPU
		gpu_labels = labels.view(len(labels))
		gpu_labels_neg = labels_neg.view(len(labels_neg))
		gpu_labels = Variable(gpu_labels.cuda())
		gpu_labels_neg = Variable(gpu_labels_neg.cuda())
        
		#print('anc gpu labels shape', gpu_labels.shape)
		#print('neg gpu labels shape', gpu_labels_neg.shape)


		# Concatenate labels for softmax/crossentropy targets
		target_softmax = torch.cat((gpu_labels, gpu_labels, gpu_labels_neg), dim=0)
		#print('target_old', target_old)
		#print('target', target_old.shape)

		target = torch.cat((labels, labels, labels_neg)).cuda()
		#torch.save(target, 'labels_batch.pt')
		#print('target', target)
		#print('target', target.shape)
        
		#print('labels',labels)
		#print('labels',labels.shape)



        
        #labels = torch.cat((gpu_labels, gpu_labels, gpu_labels_neg), dim=0)

		# Get the (e.g. hardest) triplets in this minibatch #Tripletselection!!
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, target)  #embeding_size=48 - how does it work??? labels should be target
		#print('NTRIPLETS', num_triplets)
		#print('triplet coordinates', triplets)
		#print('trip_a', (triplets[:, 0].tolist()))
		#print('trip_p',triplets[:, 1].tolist())
		#print('trip_n',triplets[:, 2].tolist())
		#print('anchors',embeddings[(triplets[:, 0].tolist())])
		#print('pos',embeddings[(triplets[:, 1].tolist())])
		#print('neg',embeddings[(triplets[:, 2].tolist())])
        

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		num_triplets = 0
		if num_triplets == 0:
			#ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			#an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
            
			cos = nn.CosineSimilarity(dim=1, eps=1e-6)
			ap_distances = 1 - cos(anchor_embed, pos_embed)
			term2 = F.relu(cos(anchor_embed, neg_embed) - 0.5)
			#an_distances = 1 - cos(anchor_embed, neg_embed)
            
			#mu = ((ap_distances.sum(0) + an_distances.sum(0)) * 0.5)/16 #16 is bath number
			#print('mu',mu)

			#print('ap_distances', ap_distances)
			#print('ap_distances/mu', ap_distances/mu)
			#print('an_distances', an_distances)
			#print('an_distances/mu', an_distances/mu)


		'''
		else:
			# Use CUDA if we can
			if anchor_embed.is_cuda: triplets = triplets.cuda()

			# Compute triplet loss over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
		'''
            
            
            
            #scale_factor
			#print('trip_a', triplets[:, 0])
			#print('trip_p',triplets[:, 1])
			#print('trip_n',triplets[:, 2])
            
			#print(ap_distances)
			#print(an_distances)
            #scaled distances
			#mu = ((ap_distances + an_distances) * 0.5)/16 #16 is bath number
			#print(mu)
			#print(mu.sum(0))
			#print(mu.sum(0)*ap_distances)
            
			#print('anc_emb',embeddings[triplets[:, 0]])
			#print(type(embeddings[triplets[:, 0]]))
			#print('anc embeddings shape', embeddings[triplets[:, 0]].shape)
			#print('pos_emb',embeddings[triplets[:, 1]])
			#print('pos embeddings shape', embeddings[triplets[:, 1]].shape)
			
                             
		
		# Compute the triplet losses
		#triplet_losses = ap_distances + (1/(10*an_distances + 2))
		triplet_losses = ap_distances + term2
		#if triplet_losses.mean()>100 or math.isnan(triplet_losses.mean()):
			#print('fcn_ap = ',ap_distances)
			#print('fcn_an = ', an_distances)
			#print('fcn_anchor_embed_vis = ',anchor_embed)
			#print('fcn_pos_embed_DNA = ',pos_embed)
			#print('fcn_neg_embed_DNA = ',neg_embed)



		#triplet_losses = ap_distances/mu + (1/(an_distances/mu))

		# Compute softmax loss		
		#loss_softmax = self.loss_fn(input=preds, target=target_softmax-1)
		loss_softmax = 1
		# Compute the total loss
		#loss_total = self.lambda_factor*triplet_losses.mean() + loss_softmax
		loss_total = triplet_losses.mean() #+ loss_softmax

		# Return them all!
		return loss_total, triplet_losses.mean(), loss_softmax
    
    
class CosineLoss(nn.Module):
	def __init__(self, triplet_selector, margin=0.0,  lambda_factor=0.01):
		super(OnlineReciprocalSoftmaxLoss, self).__init__()
		self.margin = margin
		self.loss_fn = nn.CrossEntropyLoss()
		self.lambda_factor = lambda_factor
		self.triplet_selector = triplet_selector
					
	def forward(self, anchor_embed, pos_embed, neg_embed, preds, labels, labels_neg):
        
        
        
		# Combine the embeddings from each network
		embeddings = torch.cat((anchor_embed, pos_embed, neg_embed), dim=0) #works
        
		# Define the labels as variables and put on the GPU
		gpu_labels = labels.view(len(labels))
		gpu_labels_neg = labels_neg.view(len(labels_neg))
		gpu_labels = Variable(gpu_labels.cuda())
		gpu_labels_neg = Variable(gpu_labels_neg.cuda())


		# Concatenate labels for softmax/crossentropy targets
		#target_softmax = torch.cat((gpu_labels, gpu_labels, gpu_labels_neg), dim=0)

		target = torch.cat((labels, labels, labels_neg)).cuda()

        
        #labels = torch.cat((gpu_labels, gpu_labels, gpu_labels_neg), dim=0)

		# Get the (e.g. hardest) triplets in this minibatch #Tripletselection!!
		triplets, num_triplets = self.triplet_selector.get_triplets(embeddings, target)  #embeding_size=48 - how does it work??? labels should be target
        

		# There might be no triplets selected, if so, just compute the loss over the entire
		# minibatch
		num_triplets = 0
		if num_triplets == 0:
			#ap_distances = (anchor_embed - pos_embed).pow(2).sum(1)
			#an_distances = (anchor_embed - neg_embed).pow(2).sum(1)
			cos = nn.CosineSimilarity(dim=1, eps=1e-6)
			ap_distances = 1 - cos(anchor_embed, pos_embed)
			an_distances = 1 - cos(anchor_embed, neg_embed)

            
            
			
		else:
			# Use CUDA if we can
			if anchor_embed.is_cuda: triplets = triplets.cuda()

			# Compute triplet loss over the selected triplets
			ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
			an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
		
		# Compute the triplet losses
		triplet_losses = ap_distances + (1/(an_distances + 0.1))



		#triplet_losses = ap_distances/mu + (1/(an_distances/mu))

		# Compute softmax loss		
		#loss_softmax = self.loss_fn(input=preds, target=target_softmax-1)
		loss_softmax = 1
		# Compute the total loss
		loss_total = triplet_losses.mean() #+ loss_softmax

		# Return them all!
		return loss_total, triplet_losses.mean(), loss_softmax
