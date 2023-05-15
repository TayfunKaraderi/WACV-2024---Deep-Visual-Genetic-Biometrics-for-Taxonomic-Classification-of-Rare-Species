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
from sklearn.datasets import make_blobs


# Define our own plot function
def scatter(x, labels, filename):
	# Get the number of classes (number of unique labels)
	num_classes = np.unique(labels).shape[0]

	# Choose a color palette with seaborn.
	palette = np.array(sns.color_palette("hls", num_classes+1))

	# Map the colours to different labels
	label_colours = np.array([palette[int(labels[i])] for i in range(labels.shape[0])])

	# Create our figure/plot
	f = plt.figure(figsize=(8, 8))
	#ax = plt.subplot(aspect='equal')

	# Plot the points
	#ax.scatter(	x[:,0], x[:,1], 
	#			lw=0, s=40, 
	#			c=label_colours, 
	#			marker="o")
    
	#X, y = make_blobs(centers=num_classes)
	#y =  np.unique(labels)
	ax = plt.subplot(1,1,1)
	cm = plt.get_cmap('gist_rainbow')
	ax.set_prop_cycle('color', [cm(1.*i/num_classes) for i in range(num_classes)])


	for c in  np.unique(labels):
		ax.scatter(x[labels==c,0],x[labels==c,1],label=c)

	# Do some formatting
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis('on')
	ax.axis('tight')
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
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
	embeddings = np.load('DNA_embeddings.npy')
	labels = np.load('DNA_labels.npy')

	print("Loaded embeddings")

	# Visualise the learned embedding via t-SNE
	visualiser = TSNE(n_components=2, perplexity=args.perplexity)

	# Reduce dimensionality
	reduction = visualiser.fit_transform(embeddings)

	print("Visualisation computed")

	# Plot the results and save to file
	scatter(reduction, labels, os.path.basename(args.embeddings_file)[:-4])

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
	