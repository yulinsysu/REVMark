import cv2
import torch
import torch.nn as nn
from REVMark import Encoder, Decoder, framenorm
from torchvision.utils import save_image

def load_video():
    cap = cv2.VideoCapture('dataset/1_1.mp4')
    video = []
    for i in range(8):
        ret, frame = cap.read()
        video.append(torch.from_numpy(frame[:128,:128].transpose(2,0,1).astype('float32') / 255))
    return torch.stack(video,dim=1).unsqueeze(0)

if __name__ == '__main__':
    device = 'cuda'
    encoder = Encoder(96, [8,128,128]).to(device).eval()
    decoder = Decoder(96, [8,128,128]).to(device).eval()
    encoder.load_state_dict(torch.load('checkpoints/Encoder.pth'))
    decoder.load_state_dict(torch.load('checkpoints/Decoder.pth'))

    encoder.tasblock.enable = True
    decoder.tasblock.enable = True

    cover = load_video().to(device)*2-1
    m = torch.randint(0, 2, (1,96)).float().to(device)

    r = encoder(cover, m)
    stego = (cover + 6.2*framenorm(r)).clamp(-1,1)
    noise = stego + torch.randn(*cover.shape).to(device)*0.04*2
    d = decoder(noise)

    accu = ((m >= 0.5).eq(d >= 0.5).sum().float() / m.numel()).item()
    psnr = 10 * torch.log10(4 / torch.mean((cover-stego)**2)).item()
    print('accu: ', accu, 'psnr: ', psnr)