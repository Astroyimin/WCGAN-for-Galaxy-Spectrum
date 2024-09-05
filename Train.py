import torch.nn as nn
import torch
import numpy as np
from torchsummary import summary
import load_data
from torch.utils.tensorboard import SummaryWriter
import argparseinfo
from Model import Generator,Discriminator
import time
from PLot import get_spec_plot

opt = argparseinfo.HyperParameter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU is available")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("No GPU available")


writer = SummaryWriter(log_dir="runs/training_logs")

spec,cond,wavelength = load_data.datavalue(opt.sp_size)
train_loader = load_data.prepareDataSet(spec,cond,opt.n_dat,opt.n_val,device,batchsize=opt.batch_size,shuffle=False)

generator = Generator()
discriminator = Discriminator()
generator.to(device)
discriminator.to(device)

summary(generator, [(4,),(opt.latent_dim,)])
summary(discriminator, [(4,), (opt.sp_size,)])

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

"""
    ---- Traing Main  -----
    Description :

    Author: Yimin Huang
    Model : Wasserstein condiational GAN (WCGAN)
    Sep 3th,Osaka,Japan

"""

print('\n------------<<< START TRAINING>>>------------\n')
for epoch in range(opt.n_epochs):
    batch_time =0 
    epoch_start_time = time.time()
    epoch_d_loss = 0.
    epoch_g_loss =0.
    batches_done = 0
    for i, (spec,label) in enumerate(train_loader):

        batch_start_time = time.time()
        batch_size = opt.batch_size

        # Noise
        noise = torch.randn(batch_size, opt.latent_dim)
        noise = noise.to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        fake_spec = generator(label,noise).detach()
        loss_D = -torch.mean(discriminator(label,spec))+torch.mean(discriminator(label,fake_spec))
        loss_D.backward()
        optimizer_D.step()

        epoch_d_loss += loss_D.item()

        # Clip Weight
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)
        
        if i % opt.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            gen_spec = generator(label,noise)
            loss_G = -torch.mean(discriminator(label,gen_spec))
            loss_G.backward()
            optimizer_G.step()

            epoch_g_loss += loss_G.item()
        # Store the image in training
        if batches_done % opt.sample_interval == 0:
            spectrum_image = get_spec_plot(wavelength,
                                           gen_spec[0].cpu().detach().numpy(),
                                           title=f"Spectrum_at_EPOCH{epoch}_Batch{batches_done}"
                                           )
            label_np = label.cpu().detach().numpy()
            label_np = label_np[0]
            titleinfo = f'age {label_np[0]:.1e} metallicity {label_np[1]:.1e} smass {label_np[2]:.1e} red shift {label_np[3]:.1e}'
            writer.add_image(f"BATCH_{batches_done}_{titleinfo}/EPOCH", spectrum_image, epoch, dataformats='HWC')
        batch_time += time.time() - batch_start_time
        batches_done += 1
    writer.add_scalar('Loss/D_LOSS', epoch_g_loss/opt.batch_size, epoch)
    writer.add_scalar('Loss/G_LOSS', epoch_d_loss/opt.batch_size, epoch)
    epoch_time = time.time()-epoch_start_time
    each_batch_time = batch_time/opt.batch_size
    print(
    "[Epoch %d/%d] [Epoch time: %.2f sec] [Time per sample: %.6f sec] [D loss: %f] [G loss: %f] "
    % (epoch, opt.n_epochs, epoch_time, each_batch_time,epoch_d_loss/opt.batch_size, epoch_g_loss/opt.batch_size)
    )
writer.close()
