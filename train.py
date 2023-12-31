"""
Author: Lee Taylor

file : train.py - train the C3D model from: https://arxiv.org/pdf/2206.13318v3.pdf
"""
from model import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # Init. model
    s2m = stage2model = C3D(num_classes=2)

    s2m.checkpoint_path = "checkpoints/augmented_normalized_ratiosampling_batchsize64_patch/"

    # s2m.load_checkpoint("checkpoints/augmented_normalized_ratiosampling_batchsize64/C3D_at_epoch7.pth")

    # Feed video data into lwC3D for training
    s2m.train_c3d()

    # Mark end of file
    pass
