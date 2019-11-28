import sys
import argparse
from torchsummary import summary

from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn.functional as F

from implementations.srgan.models import *
from implementations.srgan.datasets import *

if __name__ == '__main__':
    os.makedirs("images", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    read_epoch = 4
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=read_epoch, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
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

    discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
    feature_extractor = FeatureExtractor()

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        feature_extractor = feature_extractor.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_content = criterion_content.cuda()

    if opt.epoch != 0:  # 처음부터 학습이 아닐 경우에는 saved_models에서 해당 시작 위치에 해당하는 checkpoint 정보 가져오기
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/generator_%d.pth"%opt.epoch))
        discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"%opt.epoch))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))  # generator optimizer
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr,
                                   betas=(opt.b1, opt.b2))  # discriminator optimizer

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    dataloader = DataLoader(  # training data read
        ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=hr_shape,max_len=8000),# root = ../../data/img_align_celeba &  hr_shape = hr_shape
        batch_size=opt.batch_size,  # batch size ( mini-batch )
        shuffle=True,  # shuffle
        num_workers=opt.n_cpu,  # using 8 cpu threads
    )

    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor)) # low resolution
        imgs_hr = Variable(imgs["hr"].type(Tensor)) # high resolution

        #print(imgs_lr.shape)
        #print(imgs_hr.shape)
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        #print(imgs_lr.size(0)) # batch_size
        #print(*discriminator.output_shape) # (1,16,16)
        #print(valid.shape)
        #print(fake.shape)
        #exit()
        # ------------------
        #  Train Generators
        # ------------------


        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)


        #batches_done = epoch * len(dataloader) + i
        #if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
        gen_sr = make_grid(gen_hr, nrow=1, normalize=False) # change normalize=True => False
        imgs_hr = make_grid(imgs_hr, nrow=1, normalize=False)
        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=False) # normalize means that shift the image to the range(0,1), by the min and max values specified by range. Default = False
        img_grid = torch.cat((imgs_lr, gen_sr,imgs_hr), -1)
        save_image(img_grid, "test_images/%d.png" % i, normalize=False)
        if(i==10):
            exit()