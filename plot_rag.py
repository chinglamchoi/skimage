"""

- Comparing 3 segmentation algorithms: Quickshift, SLIC, Felzebszwalb
- Comparing pre & post RAG segmentations

"""
from skimage import data, color
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.future import graph
import numpy as np 
from matplotlib import pyplot as plt


def show_img(img, algo, side_by_side=False):
    width = 10.0
    height = img[0].shape[0]*width/img[0].shape[1]
    if side_by_side:
        pad = 5.0
        f = plt.figure(figsize=(width*2+pad, height))
        f.add_subplot(1,2,1)
        plt.imshow(img[0]) #plt.imshow(np.rot90(imgRr,2))
        plt.title(algo[0])
        f.add_subplot(1,2,2)
        plt.imshow(img[1])
        plt.title(algo[1])
    else:
        # just display one RGB image
        f = plt.figure(figsize=(width, height))
        plt.imshow(img)
        plt.title(algo)
    plt.show(block=True)


img = data.coffee()

labels = [quickshift(img, kernel_size=3, max_dist=6, ratio=0.5), slic(img, compactness=30, n_segments=400), felzenszwalb(img, scale=100, sigma=0.5, min_size=50)]
label_rgbs = [color.label2rgb(label, img, kind='avg') for label in labels]
algos = [["Quickshift", "SLIC (K-Means)", "Felzenszwalb"], [["Quickshift Before RAG", "Quickshift After RAG"], ["SLIC (K-Means) Before RAG", "SLIC (K-Means) After RAG"], ["Felzenszwalb Before RAG", "Felzenszwalb After RAG"]]]

rags = [graph.rag_mean_color(img, label) for label in labels]
edges_drawn_all = [plt.colorbar(graph.show_rag(label, rag, img)).set_label(algo) for label, rag, algo in zip(labels, rags, algos[0])]

for edge_drawn in edges_drawn_all:
    plt.show()

# only display edges with weight > thresh
final_labels = [graph.cut_threshold(label, rag, 29) for label, rag in zip(labels, rags)]
final_label_rgbs = [color.label2rgb(final_label, img, kind='avg') for final_label in final_labels]

for label_rgb, final_label_rgb, algo in zip(label_rgbs, final_label_rgbs, algos[1]):
    show_img((label_rgb, final_label_rgb), algo, side_by_side=True)
