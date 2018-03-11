import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from datetime import datetime

from torch.autograd import Variable
from tqdm import tqdm, trange

import models

parser = argparse.ArgumentParser(description='PyTorch Wasserstein GAN Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='results dir')
parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')

parser.add_argument('--dataset', metavar='DATASET', default='cifar10', help='dataset name')
parser.add_argument('--dataset-path', metavar='DATASET_PATH', default='./dataset', help='dataset folder')

parser.add_argument('--input-size', type=int, default=32, help='image input size')
parser.add_argument('--channels', type=int, default=3, help='input image channels')
parser.add_argument('--z-size', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--gen-filters', type=int, default=64)
parser.add_argument('--disc-filters', type=int, default=64)
parser.add_argument('--clamp', type=float, default=0.01, help='value to clamp weights on')
parser.add_argument('--disc-iters', type=int, default=30,
                    help='number of discriminator iterations per each generator iteration')
parser.add_argument('--n_extra_layers', type=int, default=0,
                    help='Number of extra layers for generator and discriminator')

parser.add_argument('--gpus', default='0', help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lrg', '--learning-rate-gen', default=5e-5, type=float, metavar='LRG',
                    help='initial learning rate for generator')
parser.add_argument('--lrd', '--learning-rate-disc', default=5e-5, type=float, metavar='LRD',
                    help='initial learning rate for discriminator')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int, metavar='N', help='print frequency (default: 50)')

parser.add_argument('--seed', default=42, type=int, help='random seed (default: 42)')


def main():
    args = parser.parse_args()

    # random.seed(args.manualSeed)
    torch.manual_seed(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # if args.evaluate:
    #     args.results_dir = '/tmp'
    if args.save is '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)

    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root=args.dataset_path, download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(args.input_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    else:
        return
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.workers)

    netG = models.WGANGenerator()
    print(netG)
    netD = models.WGANDiscriminator()
    print(netD)

    fixed_noise = torch.FloatTensor(args.batch_size, args.z_size, 1, 1).normal_(0, 1)

    args.gpus = [int(i) for i in args.gpus.split(',')]
    torch.cuda.set_device(args.gpus[0])
    cudnn.benchmark = True
    netG = torch.nn.DataParallel(netG, args.gpus)
    netD = torch.nn.DataParallel(netD, args.gpus)

    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lrd)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lrg)

    for epoch in trange(args.epochs):
        disc_iter = 200 if epoch == 0 else args.disc_iters
        for i, (real_inputs, _) in enumerate(tqdm(dataloader)):
            # Train Discriminator
            # clamp parameters to a cube
            for p in netD.parameters():
                p.requires_grad = True
                p.data.clamp_(-args.clamp, args.clamp)
            optimizerD.zero_grad()

            # real data
            real_inputs = Variable(real_inputs.cuda())
            errD_real = -netD(real_inputs)
            optimizerD.step()

            # fake data
            noise = Variable(torch.FloatTensor(real_inputs.shape[0], args.z_size, 1, 1).normal_(0, 1))
            # with torch.no_grad(): #TODO
            fake_inputs = Variable(netG(noise).data)

            errD_fake = netD(fake_inputs)

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            # Train Generator
            if i % disc_iter:
                optimizerG.zero_grad()
                noise = Variable(torch.FloatTensor(args.batch_size, args.z_size, 1, 1).normal_(0, 1))
                fake = netG(noise)
                errG = -netD(fake)
                errG.backward()
                optimizerG.step()
            else:
                noise = Variable(torch.FloatTensor(args.batch_size, args.z_size, 1, 1).normal_(0, 1))
                fake = netG(noise)
                errG = netD(fake)

            if i % args.print_freq == 0:
                tqdm.write('[{}/{}][{}/{}] Loss_D: {} Loss_G: {} '
                           'Loss_D_real: {} Loss_D_fake {}'.format(epoch, args.epochs, i, len(dataloader),
                                                                   errD.data.mean(), errG.data.mean(),
                                                                   errD_real.data.mean(), errD_fake.data.mean()))

        real_inputs = real_inputs.mul(0.5).add(0.5)
        vutils.save_image(real_inputs.data, '{0}/real_samples.png'.format(save_path))
        # with torch.no_grad(): #TODO
        fake = netG(Variable(fixed_noise))
        fake.data = fake.data.mul(0.5).add(0.5)
        vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(save_path, epoch))


if __name__ == '__main__':
    main()
