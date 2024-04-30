'''
REVMark训练的参考程序，默认情况下编解码器不包含TAsBlock，根据需要自定义DataSet类
'''
import cv2
import torch
import torch.nn as nn
import argparse
from REVMark import Encoder, Decoder, Discriminator, framenorm
from DiffH264 import DiffH264


def weight_init(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in')
        if m.bias is not None: m.bias.data.zero_()


def train():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = 'kinetics400/train/'
    args.device = 'cuda'
    args.epochs = 50
    args.lr = 0.0001
    args.lr2 = 0.00001
    args.batch_size = 16
    args.msg_size = 96
    args.log_step = 100
    args.save_step = 5000
    
    trainloader = torch.utils.data.DataLoader(DataSet(args.dataset), batch_size=args.batch_size, shuffle=True, drop_last=True)

    encoder = Encoder(args.msg_size, [8,128,128]).to(args.device)
    decoder = Decoder(args.msg_size, [8,128,128]).to(args.device)
    discriminator = Discriminator().to(args.device)
    compressor = DiffH264([8,128,128], [1.5,5], args.device)

    encoder.apply(weight_init)
    decoder.apply(weight_init)
    discriminator.apply(weight_init)

    msgloss = nn.BCELoss()
    optimizer_coder = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=args.lr)
    optimizer_discr = torch.optim.Adam(discriminator.parameters(), lr=args.lr2)

    print("start training.")
    step = 0
    for epoch in range(args.epochs):
        for data in trainloader:
            lambda1 = 1
            if step < 1500: alpha = 20
            elif step < 2000: alpha = 20 - 13.8/500*(step-1500)
            else: alpha = 6.2
            if step < 3000: lambda2 = 0
            elif step < 10000: lambda2 = 0.0005/7000*(step-3000)
            else: lambda2 = 0.0005
            if step < 2000: lambda3 = 0
            elif step < 4000: lambda3 = 10/2000*(step-2000)
            else: lambda3 = 10
            if step < 3000: compress_flag = False
            else: compress_flag = True
            
            cover = data.to(args.device)
            origin_msg = torch.randint(0, 2, (args.batch_size,args.msg_size)).float().to(args.device)

            residual = encoder(cover, origin_msg)
            residual = alpha*framenorm(residual)
            stego = (cover + residual).clamp(-1,1)

            if compress_flag:
                noise = compressor.compress(stego)
            else:
                noise = stego
            extract_msg = decoder(noise)

            msg_loss = msgloss(extract_msg, origin_msg)
            mask_loss = maskloss(stego, cover, alpha)
            adv_loss = discriminator(stego)
            
            loss = lambda1*msg_loss + lambda2*adv_loss + lambda3*mask_loss
            optimizer_coder.zero_grad()
            loss.backward()
            optimizer_coder.step()
            
            if lambda2 != 0:
                d_loss = discriminator(cover) - discriminator(stego.detach())
                optimizer_discr.zero_grad()
                d_loss.backward()
                optimizer_discr.step()
            
            step += 1
            if step % args.log_step == 0:
                accu = ((extract_msg >= 0.5).eq(origin_msg >= 0.5).sum().float() / origin_msg.numel()).item()
                psnr = 10 * torch.log10(4 / torch.mean((cover-stego)**2)).item()
                print('step:', step, 'accu:', accu, 'psnr:', psnr)
                if step >= 1500 and accu < 0.8:
                    step = 1
                    encoder.apply(weight_init)
                    decoder.apply(weight_init)
                    optimizer_coder = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=args.lr)
            if step % args.save_step == 0:
                torch.save(encoder.state_dict(), 'checkpoints/revmark-encoder.pth')
                torch.save(decoder.state_dict(), 'checkpoints/revmark-decoder.pth')
                torch.save(discriminator.state_dict(), 'checkpoints/revmark-discriminator.pth')
            if step % 30000 == 0:
                for param_group in optimizer_coder.param_groups:
                    param_group['lr'] *= 0.5
            if step > 60000:
                exit()
        print('epoch:', epoch)


if __name__ == '__main__':
    train()