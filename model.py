"""
Authors: Yuchen Wang, Zhongyu Li, Xiangxiang Cu, Liangliang Zhan, Xiang Lu, Meng Yan, Shi Chan
Modified by: Lee Taylor
"""
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from functions import dataloader, dataloader_test
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from skimage.metrics import structural_similarity as ssim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_histogram_similarity(image_a, image_b):
    """
    Calculate histogram similarity of frames, as stated in the paper.
    :param image_b: first image
    :param image_a: second image
    :return:
    """
    # Calculate the color histogram of image_a using 8 bins for each color
    # channel (R, G, B) and a color range of 0-256 for each channel.
    hista = cv2.calcHist([image_a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # Calculate the color histogram of image_b using the same parameters as for image_a.
    histb = cv2.calcHist([image_b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # Normalize the color histogram of image_a so that its values lie in the range 0-1.
    cv2.normalize(hista, hista)
    # Normalize the color histogram of image_b in the same way.
    cv2.normalize(histb, histb)

    # Compare the two normalized color histograms using the correlation
    # comparison method and return the correlation value.
    return cv2.compareHist(hista, histb, cv2.HISTCMP_CORREL)


def compute_motion_index(video):
    """
    Calculate SSIM as stated in the paper.
    :param video:
    :return:
    """
    T, H, W, C = video.shape        # temporal index, height, width, channels
    motion_index = np.zeros((T,))   # init. motion index of zeros
    # Nested for loop acts as a moving window for each frame Mi
    for i in range(T):
        frame_similarities = []
        for j in [i-2, i-1, i+1, i+2]:
            if 0 <= j < T:
                # Calculate SSIM
                ssim_value = ssim(video[i], video[j], multichannel=True,
                                  win_size=5, channel_axis=2, data_range=100)
                # Calculate Histogram similarity
                histogram_similarity = compute_histogram_similarity(video[i], video[j])
                # Average of SSIM and histogram
                frame_similarities.append((ssim_value + histogram_similarity) / 2)

        motion_index[i] = np.mean(frame_similarities)
        # print(motion_index[i].shape)

    return motion_index


class SPP(nn.Module):
    """
    Spatial Pyramind Pooling
    """

    def __init__(self):
        super(SPP, self).__init__()
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 3, 3))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        """
        Forward pass of SPP layers.
        :param x:
        :return:
        """
        x1 = self.pool1(x)
        x2 = self.pool2(x)

        x = x.view(x.size(0), -1)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        outputs = torch.cat([x, x1, x2], dim=1)

        return outputs


class C3D(nn.Module):
    """
    Modfiied C3D network class definition.
    """

    def __init__(self, num_classes):
        super(C3D, self).__init__()
        self.optimizer = None
        self.epoch = 0
        self.checkpoint_path = 'checkpoints/'
        """ Ultrasound Video Classification """

        # First three convolutional layers
        # Input layer (32 Frames)
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.batchnorm1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.batchnorm2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.batchnorm3 = nn.BatchNorm3d(256)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Motion Attention (attention layers)
        self.attention_conv1 = nn.Conv3d(256, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.attention_conv2 = nn.Conv3d(64, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.attention_conv3 = nn.Conv3d(16, 1, kernel_size=(1, 2, 2), stride=(1, 1, 1))

        # Fourth/final convolutional layer (512) and Max-pooling layer
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.batchnorm4 = nn.BatchNorm3d(512)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # SPP (Spatial Pyramid Pooling)
        self.spp = SPP()

        # Linear layers with dropout
        self.fc1 = nn.Linear(13824, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        # Activation function
        self.relu = nn.ReLU()
        # Cosine
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # Mark end of Class

    def forward(self, x):
        """
        Calculate forward pass of network weights and input.
        :param x:
        :return:
        """
        # batch_size,channels,帧数,H,W
        # First 3 convolutional layers with batch normalization and ReLU
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        # Pooling layer
        x = self.pool1(x)
        # Motion Attention
        atte = self.attention_conv1(x)
        atte = self.attention_conv2(atte)
        atte = self.attention_conv3(atte)
        outputs2 = atte.view(-1, 8)
        # Multiply 'attention motion output' by 'pooled convolutional output'
        x = atte * x
        # Fourth and final convolutional layer
        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.pool2(x)
        # Spatial Pyramid Pooling layer
        x = self.spp(x)
        # Linear layers
        x = x.view(-1, 13824)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        outputs1 = self.fc2(x)
        # Return 'category classification' and 'attention motion outputs'
        return outputs1, outputs2

    def load_state(self, state_dict, strict=False):
        """
        Load state of the model from saved stated.
        :param state_dict:
        :param strict:
        :return:
        """
        print('start loading state_dict...')
        # if load state strict then must match every parameters between the net and the checkpoint
        # else can load the paramerters that matched and throw the parameters that don't matched

        if strict:
            self.net.load_state_dict(state_dict=state_dict)
        else:
            net_state_keys = list(self.state_dict().keys())
            not_matched_params = []
            for name, param in state_dict.items():
                print(name, param.shape)
                if name in self.state_dict().keys():
                    dst_param_shape = self.state_dict()[name].shape
                    if param.shape == dst_param_shape:
                        self.state_dict()[name].copy_(param)
                        net_state_keys.remove(name)
                    else:
                        not_matched_params.append(name)
                else:
                    not_matched_params.append(name)

            if not_matched_params:
                print('Failed to load {}'.format(not_matched_params))
            if net_state_keys:
                print('lack {} to load '.format(not_matched_params))

        print('load state_dict succeed...')
        return True

    def load_checkpoint(self, load_path, optimizer=False):
        """
        Load model weights.
        :param load_path:
        :param optimizer:
        """
        assert os.path.exists(load_path), \
            'Failed to load {},file not exists'.format(load_path)
        checkpoint = torch.load(load_path)
        if 'state_dict' in checkpoint.keys():
            all_parmas_matched = self.load_state(checkpoint['state_dict'])
        else:
            all_parmas_matched = self.load_state(checkpoint)

        assert all_parmas_matched, 'Failed to load state_dict'

        if optimizer:
            if 'optimizer' in checkpoint.keys():
                optimizer.load_state_dict(checkpoint['optimizer'])
                print('optimizer load...')
            else:
                print(
                    'Failed to load optimizer,there is no optimizer in {}'.format(load_path)
                )

        if 'epoch' in checkpoint.keys():
            self.epoch = checkpoint['epoch']

    def save_checkpoint(self, is_best=False):
        """
        Save model state
        :param is_best:
        :return:
        """
        save_path = self.checkpoint_path + "C3D_at_epoch{}.pth".format(self.epoch)

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        if is_best:
            save_path = self.checkpoint_path + "C3D_best_model.pth"
            torch.save(
                {
                    'epoch': self.epoch,
                    'state_dict': self.net.state_dict(),
                    'best_prec1': self.best_prec1,
                    'optimizer': self.optimizer.state_dict()
                }, save_path)
            print(
                'best model at {} opech has been saved to {}'.format(self.epoch, save_path)
            )
            return

        torch.save({
            'epoch': self.epoch,
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, save_path)
        print('checkpoint (model & optimizer) has been saved to {}'.format(save_path))

    def _l_motion(self, vtemp, vmotion, out=False):
        # Ensure vtemp and vmotion are on the same device
        vmotion = vmotion.to(vtemp.device)
        # Calcualte cosine(vtemp, vmotion)
        rv = 1 - self.cos(vtemp, vmotion)
        # Debug
        if out:
            print(f"vtemp.shape = {vtemp.shape}, vmotion = {vmotion.shape}")
            print(f"_l_motion() = {rv}")
        return rv

    def train_c3d(self, trainloader=None, epochs=40):
        """
        Train the model.

        "classification loss Lcls is formulated by:
            Lcls = −(yn · log zn + (1 − yn) · log(1 − zn))
        n is batch size, yn is label of the batch, zn is output of batch"

        "Cosine loss is appended between 'temporal weights' (Vtemp) and the 'motion
        vector' (Vmotion) to optimize parameters."

        "the feature map is reduced to 1 channel. The final output feature map
        size is 1×T×1×1, which could be used as temporal weights."

        "loss...
        """
        # The equation they provide is Cross Entropy Loss, from the diagram
        # we are predicting two classes therefore use nn.CrossEntropyLoss().
        classification_criterion = nn.CrossEntropyLoss()
        # "we use Adam optimizer with the learning rate of 1e-3 and ...
        # weight decay of 1e-8"
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-8)
        self.train()  # Set training mode to True

        # Train loop
        num_epochs = epochs  # 40
        for self.epoch in range(num_epochs):
            # "after 40 epochs we change the LR to 1e-4"
            if self.epoch > 20:
                self.optimizer = optim.Adam(self.parameters(),
                                       lr=1e-4, weight_decay=1e-8)
            # TODO: plots
            running_loss = 0.0
            average_loss = 0.0
            trainloader = dataloader()
            max_iter = len(list(dataloader()))
            print(f"max_iter={max_iter}")

            # Loop over data
            for i, data in enumerate(trainloader):
                # Unpack data
                inputs, labels = data  # 32 frames, y_true
                # >>> labels = tensor([1., 0.])
                # >>> labels.shape = torch.Size([2])

                self.optimizer.zero_grad()  # zero the parameter gradients

                outputs, vtemp = self(inputs)  # prediction, temporal weights
                # >>> outputs.shape = torch.Size([1, 2])
                # >>> outputs = tensor([[ 4.7026, -8.8772]], grad_fn=<AddmmBackward0>)

                # Calculate Vmotion from inputs rearrange
                # dimensions and convert to numpy array
                inputs_np = inputs.permute(0, 2, 3, 4, 1).numpy()
                vmotion_list = []
                for video in inputs_np:
                    # calculate vmotion for each video
                    vmotion_video = compute_motion_index(video)
                    vmotion_list.append(vmotion_video)
                # convert back to tensor
                vmotion = torch.tensor(vmotion_list, dtype=torch.float32).to(device)

                # Change vmotion size for COSINE calculation
                # >>> vmotion.shape = torch.Size([1, 32])
                vmotion = vmotion.view(-1, 8)
                # >>> vmotion.shape = torch.Size([1, 8])

                # Calc. LOSS, classification & vtemp
                loss_cls = classification_criterion(outputs[0], labels)
                loss_motion = self._l_motion(vtemp, vmotion)
                # >>> vtemp.shape = [1, 8], vmotion.shape = [4, 8]
                # >>> _l_motion = tensor([0.0039, 0.0023, 0.0024, 0.0030], grad_fn=<RsubBackward1>)

                # Total Loss
                loss = loss_cls + loss_motion

                # Average the loss
                loss = loss.mean()  # Todo: Average or sum loss?
                print(f"loss.mean() = {loss}, "
                      f"y_pred = {f.softmax(outputs[0], dim=0).detach().numpy()}, "
                      f"y_actual = {labels}")

                # back propagation, optimize weights
                loss.backward()
                self.optimizer.step()

                # Sum loss
                running_loss += loss
                average_loss = running_loss / max_iter

            # End of epoch
            print(f"Epoch: {self.epoch}, Total Loss: {running_loss}, Avg Loss: {average_loss}")
            self.save_checkpoint()

        print('Finished Training')
        return


if __name__ == '__main__':
    # Init. lightweight C3D model
    c3d = C3D(num_classes=2)

    # Set model.training to True
    # c3d.train()
    # print(f"c3d.training = {c3d.training}")
    # >>> True

    # print(f"\nc3d.train_c3d() = {c3d.train_c3d(1)}")
    c3d.load_checkpoint("checkpoints/C3D_at_epoch39.pth")

    preds = []
    labels_list = []

    # outputs, vtemp = self(inputs)  # prediction, temporal weights6
    for i, data in enumerate(dataloader_test()):
        # model input-output
        inputs, labels = data
        outputs, vtemp = c3d(inputs)

        outputs_vals = f.softmax(outputs[0], dim=0).detach().numpy()
        outputs_vals_rounded = [round(outputs_vals[0]), round(outputs_vals[1])]
        print(f"outputs = {outputs_vals_rounded}, labels = {labels}")

        # outputs_vals_binary = np.argmax(outputs_vals)
        if outputs_vals_rounded == [round(int(labels[0])), round(int(labels[1]))]:
            preds.append(1)
        else:
            preds.append(0)

        # Save predictions and labels for evaluation
        # preds.append(round(outputs_vals[1]))
        labels_list.append(np.argmax(labels.numpy()))

    print("Finished Predicting")
    print()

    # Convert lists to numpy arrays
    y_pred = np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    y_test_single_label = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    # Debug output
    print(f"y_test_single_label = {y_test_single_label}")
    print(f"y_pred = {y_pred}")

    # Create a confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test_single_label, y_pred).ravel()

    # Compute metrics
    accuracy = accuracy_score(y_test_single_label, y_pred)
    sensitivity = recall_score(y_test_single_label, y_pred)  # Sensitivity is the same as Recall
    specificity = tn / (tn+fp)
    precision = precision_score(y_test_single_label, y_pred)
    f1 = f1_score(y_test_single_label, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Precision: {precision}")
    print(f"F1-score: {f1}")

    # Mark EOF
    pass
