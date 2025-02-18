"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""
''' # 해당 py파일에서 사용하지 않는 import  
import os
import numpy as np
import math
import itertools
import torchvision.transforms as transforms
import torch.nn as nn
import torch
'''

import sys
import argparse
from torchsummary import summary

from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn.functional as F

from implementations.srgan.models import *
from implementations.srgan.datasets import *
import time
if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    read_epoch = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=read_epoch, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    opt = parser.parse_args()
    print(opt)

    cuda = torch.cuda.is_available()

    hr_shape = (opt.hr_height, opt.hr_width)

    # Initialize generator and discriminator
    generator = GeneratorResNet()
    discriminator = Discriminator_withDense(input_shape=(opt.channels, *hr_shape))
    #discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
    feature_extractor = FeatureExtractor()

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    #BCE (h(x),y) = -ylog[h(x)] - (1-y)log[1-h(x)]
    # y = 0일 경우
    #BCE = -log[1-h(x)]
    # y = 1일 경우
    #BCE = -log[h(x)]
    #adversarial_GAN = torch.nn.BCEWithLogitsLoss()
    criterion_GAN = torch.nn.BCELoss()
    #criterion_GAN = torch.nn.BCEWithLogitsLoss()
    criterion_content = torch.nn.MSELoss()
    #criterion_content = torch.nn.L1Loss() # 각 원소별 차이의 절댓값을 계산

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        feature_extractor = feature_extractor.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_content = criterion_content.cuda()

    if opt.epoch != 0:  # 처음부터 학습이 아닐 경우에는 saved_models에서 해당 시작 위치에 해당하는 checkpoint 정보 가져오기
        # Load pretrained models
        #generator.load_state_dict(torch.load("saved_models/generator_%d.pth"%opt.epoch))
        #discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"%opt.epoch))
        generator.load_state_dict(torch.load("saved_models_new/generator_%d.pth" % opt.epoch))
        discriminator.load_state_dict(torch.load("saved_models_new/discriminator_%d.pth" % opt.epoch))
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))  # generator optimizer
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr,
                                   betas=(opt.b1, opt.b2))  # discriminator optimizer

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    dataloader = DataLoader(  # training data read
        ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=hr_shape,max_len=6000),# root = ../../data/img_align_celeba &  hr_shape = hr_shape
        batch_size=opt.batch_size,  # batch size ( mini-batch )
        shuffle=True,  # shuffle
        num_workers=opt.n_cpu,  # using 8 cpu threads
    )
    '''
    print("Generator Model Summary")
    summary(generator,input_size=(3,256//4,256//4))
    print("Discriminator Model Summary")
    summary(discriminator,input_size=(3,256,256))
    print("VGG19 Model Summary")
    summary(feature_extractor,input_size=(3,256,256))
    '''
    # ----------
    #  Training
    # ----------
    for epoch in range(opt.epoch, opt.n_epochs): # epoch ~ 200
        start = time.time()  # 시작 시간 저장
        for i, imgs in enumerate(dataloader):
            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor)) # low resolution
            imgs_hr = Variable(imgs["hr"].type(Tensor)) # high resolution

            #print(imgs_lr.shape)
            #print(imgs_hr.shape)
            # Adversarial ground truths
            #여기서부터 공부해서 차원 맞추는 공부 할 것!!!
            #print("valid shape : {}, {} , {}".format(np.ones((imgs_lr.size(0),*discriminator.output_shape)).shape,imgs_lr.size(0),*discriminator.output_shape))
            # batch_size , 1 , 16 , 16
            #valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            #fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), 1))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), 1))), requires_grad=False)

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)


            # ---------------------
            #  Train Discriminator
            # ---------------------
            #변화량을 0로 초기화
            optimizer_D.zero_grad()

            # Loss of real and fake images
            #loss_real = adversarial_GAN(discriminator(imgs_hr), valid)
            #loss_fake = adversarial_GAN(discriminator(gen_hr.detach()), fake)
            loss_real = criterion_GAN(discriminator(imgs_hr), valid) # real loss는 1에 가깝게 (maximum)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake) # fake loss는 0에 가깝게 (minmum)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            #backpropagation
            loss_D.backward()

            #update
            optimizer_D.step()

            # ------------------
            #  Train Generators
            # ------------------
            # 변화량을 0로 초기화
            optimizer_G.zero_grad()

            # Adversarial loss
            #print("discriminator shape : ", discriminator(gen_hr).shape)
            #loss_GAN = adversarial_GAN(discriminator(gen_hr), valid)
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)
            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())
            #loss_content = criterion_GAN(gen_features, real_features.detach())
            # MSE loss function with feature extract input data (batch size, 1 , 16, 16 )

            #-torch.mean(torch.log(
            #    F.sigmoid(pos_score - neg_scores)), 0)  # BPR loss

            # Total loss
            #loss_G = loss_content + 1e-3 * loss_GAN
            loss_G = loss_content + 1e-3 *loss_GAN # Feature extract를 사용하지 않고 학습

            loss_G.backward()
            #loss_G.backward(retain_graph=True)
            optimizer_G.step()



            # --------------
            #  Log Progress
            # --------------

            batches_done = epoch * len(dataloader) + i
            if batches_done % 10 == 0:
                sys.stdout.write(
                    "[Epoch %d/%d] [Batch %d/%d][D loss: %f] [G loss: %f] [read %d images time : %dsec]\n"
                    % (epoch, opt.n_epochs, i , len(dataloader), loss_D.item(), loss_G.item(),
                       (batches_done * opt.batch_size), (time.time() - start))
                )
                start = time.time()  # 시작 시간 저장

            if batches_done % (opt.sample_interval*10) == 0:
                # UnNormalize function
                unorm = UnNormalize()
                imgs_lr = unorm(imgs_lr)
                imgs_hr = unorm(imgs_hr)
                gen_hr = unorm(gen_hr)

                # Save image grid with upsampled inputs and SRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                # 'nearest' ( default ) , 'linear' , 'bilinear' , 'bicubic', 'trilinear' , 'area'
                gen_sr = make_grid(gen_hr, nrow=1, normalize=False) # change normalize=True => False
                imgs_hr = make_grid(imgs_hr, nrow=1, normalize=False)
                imgs_lr = make_grid(imgs_lr, nrow=1, normalize=False) # normalize means that shift the image to the range(0,1), by the min and max values specified by range. Default = False
                img_grid = torch.cat((imgs_lr, gen_sr,imgs_hr), -1)
                save_image(img_grid, "images_new/%d.png" % batches_done, normalize=False)

        if epoch % opt.checkpoint_interval == 0 :
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models_new/generator_%d.pth" % (epoch+1))
            torch.save(discriminator.state_dict(), "saved_models_new/discriminator_%d.pth" % (epoch+1))
