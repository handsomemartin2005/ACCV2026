import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools

class ConfusionMatrix(object):


    def __init__(self, matrix):
        self.matrix = matrix

    def plot(self):
        self.plot_confusion_matrix()

    def plot_confusion_matrix(self):
        matrix = self.matrix
        cmap = plt.cm.YlGnBu  # 绘制的颜色 plt.cm.Blues viridis_r

        plt.figure(figsize=(10, 10))
        plt.imshow(matrix, interpolation='nearest', cmap=cmap)

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.3)
        plt.savefig('/home/robot/Projects/mamba-detr/output/COP_adj_matrix.jpg', dpi=1200, bbox_inches='tight', pad_inches=0.1)
        # plt.show()


# COP
# adj_matrix_path = '/home/robot/Projects/mamba-detr/configs/dataset/coco/adj_with_embedding/adj_matrix.pkl'
# matrix_COP = torch.load(adj_matrix_path)

# Random initialization matrix
# matrix_random = np.random.rand(17, 17)

# learnable random matirx
# matrix_random = np.random.rand(17, 17)
# learnable_matrix_random = torch.tensor(matrix_random, dtype=torch.float32, requires_grad=True)

# learnable COP matirx
adj_matrix_path = '../Graph-MDETR/dataset/coco_uav/adj_with_embedding/adj_matrix.pkl'
matrix_COP = torch.load(adj_matrix_path)

matrix_random = np.random.rand(17, 17)
np.fill_diagonal(matrix_random, 1)
learnable_matrix_random = torch.tensor(matrix_random, dtype=torch.float32, requires_grad=True)
earnable_COP_matirx = matrix_COP * learnable_matrix_random

#
cnfusionMatrix = ConfusionMatrix(matrix=learnable_matrix_random.detach().numpy())
cnfusionMatrix.plot()