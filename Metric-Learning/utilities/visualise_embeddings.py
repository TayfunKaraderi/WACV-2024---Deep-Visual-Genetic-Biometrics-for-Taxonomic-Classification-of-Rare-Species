# Core libraries
import os
import sys
import cv2
import argparse
import numpy as np

# Matplotlib / TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects

# Define our own plot function
def scatter(x, labels, filename):
	# Get the number of classes (number of unique labels)
    labels = labels[1:-1]
    x = x[1:-1]
    num_classes = np.unique(labels).shape[0]

	# Choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", num_classes+1))

	# Map the colours to different labels
    label_colours = np.array([palette[int(labels[i])] for i in range(labels.shape[0])])
    #print(label_colours.shape)
    #print(labels)
    
    '''
	# Create our figure/plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

	# Plot the points
    p = ax.scatter(	x[:,0], x[:,1], 
				lw=0, s=40, 
				c=labels, 
				marker="o",
                cmap=plt.cm.get_cmap('viridis', 25))

	# Do some formatting
    plt.colorbar(p, ticks=range(1,26), label='class')
    '''
    # Create our figure/plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

	# Plot the points
    ax.scatter(	x[:,0], x[:,1], 
				lw=0, s=40, 
				c=label_colours, 
				marker="o")

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    #ax.axis('off')
    ax.axis('tight')
    plt.tight_layout()

	# Save it to file
    plt.savefig(filename+".pdf")

# Load and visualise embeddings via t-SNE
def plotEmbeddings(args):
	# Ensure there's something there
	if not os.path.exists(args.embeddings_file):
		print(f"No embeddings file at path: {args.embeddings_file}, exiting.")
		sys.exit(1)

	# Load the embeddings into memory
	embeddings1 = np.load(args.embeddings_file + '/train_embeddings.npz')
	embeddings2 = np.load(args.embeddings_file + '/test_embeddings.npz')
    
	embeddings = np.concatenate((embeddings1['embeddings'], embeddings2['embeddings']), axis=0)

	#print(embeddings)

	print("Loaded embeddings")

	# Visualise the learned embedding via t-SNE
	visualiser = TSNE(n_components=2, perplexity=args.perplexity)

	# Reduce dimensionality
	#reduction = visualiser.fit_transform(embeddings['embeddings'])
	reduction = visualiser.fit_transform(embeddings)
    
	print("Visualisation computed")

    
	reduction_train = reduction[0:27394,:]
	reduction_test = reduction[27394:,:]

	#print(len(reduction_train), len(reduction_test))


	print(reduction)
	print(reduction.shape)
	print(reduction_train.shape, len(embeddings1['labels']))
	print(reduction_test.shape, len(embeddings2['labels']))


	# Plot the results and save to file
	scatter(reduction_train, embeddings1['labels'], os.path.basename(args.embeddings_file) + '/train_embeddings')
	scatter(reduction_test, embeddings2['labels'], os.path.basename(args.embeddings_file) + '/test_embeddings')
	#scatter(reduction_train, embeddings1['labels'], '/content/drive/MyDrive/foram-MetricLearningIdentification-master/output/full_data_rotation_augmented/train_embeddings')
	#scatter(reduction_test, embeddings2['labels'], '/content/drive/MyDrive/foram-MetricLearningIdentification-master/output/full_data_rotation_augmented/test_embeddings')

    


# Main/entry method
if __name__ == '__main__':
	# Collate command line arguments
	parser = argparse.ArgumentParser(description='Parameters for visualising the embeddings via TSNE')
	parser.add_argument('--embeddings_file', type=str, required=True,
						help="Path to embeddings .npz file you want to visalise")
	parser.add_argument('--perplexity', type=int, default=30,
						help="Perplexity parameter for t-SNE, consider values between 5 and 50")
	args = parser.parse_args()

	# Let's plot!
	plotEmbeddings(args)
	