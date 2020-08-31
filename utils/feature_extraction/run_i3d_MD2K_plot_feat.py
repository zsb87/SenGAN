# run on laptop

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from extract_features_cpu import run

def save_feat():
    run(mode="flow", 
        # load_model="models/flow_charades.pt",
        load_model="models/flow_imagenet.pt",
        sample_mode="resize",
        frequency=16,
        input_dir=flo_folder, 
        output_dir=out_folder,
        batch_size=16,
        usezip=0)


def load_feat(file):
	b = np.load(file)
	print(b["feature"])
	print(b["feature"].shape)
	print(b["frame_cnt"])
	print(b["video_name"])
	X = b["feature"]
	# X_embedded = TSNE(n_components=2).fit_transform(X)
	# print(X_embedded.shape)
	plot_tsne(X)


def plot_tsne(X):
	X_embedded = TSNE(n_components=3).fit_transform(X[0])
	c = np.zeros_like(X_embedded[:,0])
	print(c.shape)
	c[:int(c.shape[0]/2)] = 1

	ax = plt.figure(figsize=(16,10)).gca(projection='3d')
	ax.scatter(
	    xs=X_embedded[:,0], 
	    ys=X_embedded[:,1], 
	    zs=X_embedded[:,2], 
	    c=c, #class 
	    cmap='tab10'
	)
	plt.show()

if __name__ == '__main__':
	# run on laptop
    flo_folder = "../../../dataset/MD2K/"
    out_folder = "../../../dataset/MD2K/feat"

    # save_feat()
    load_feat(("../../../dataset/MD2K/GP020550/feat/GP020550-flow.npz"))

    # ffmpeg -i "GP020550.MP4" vid1/img/img_%05d.jpg

