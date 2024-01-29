import torch
from dctquantify import DCTQuantify
from MotionEstimation import ME_Spynet, flow_warp

class DiffH264():
    def __init__(self, videoshape, factors, device):
        self.framenum = videoshape[0]
        self.factors = factors
        self.net_dctq = DCTQuantify(videoshape[1], videoshape[2], color_space='yuv').to(device)
        self.motion_estimate = ME_Spynet(levelnum=4).to(device)
        self.motion_estimate.load('ME_Spynet_Full.pth')
        self.motion_estimate.eval()
        self.warp = flow_warp
        self.rgb2yuv = torch.Tensor([
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312]]).T.to(device)
        self.yuv2rgb = torch.Tensor([[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]]).T.to(device)
    
    def compress(self, video): # video:(B,C,F,H,W)
        ref_idx = torch.randint(0, self.framenum, ())
        video_yuv = torch.matmul(video[:,[2,1,0]].transpose(4,1), self.rgb2yuv).transpose(4,1) # bgr to yuv
        frameref = self.net_dctq(video_yuv[:,:,ref_idx], factor=self.factors[0]) # intra compression
        vidrec = []
        for f in range(self.framenum):
            if f == ref_idx:
                vidrec.append(frameref)
                continue
            mv = self.motion_estimate(video_yuv[:,:,f]/2+0.5, frameref/2+0.5)
            prediction = self.warp(frameref, mv, self.motion_estimate.coords)
            resirec = self.net_dctq(video_yuv[:,:,f]-prediction, factor=self.factors[1])
            vidrec.append(prediction + resirec)
        vidrec = torch.stack(vidrec, dim=2)
        video_bgr = torch.matmul(vidrec.transpose(4,1), self.yuv2rgb).transpose(4,1)[:,[2,1,0]] # yuv to bgr
        return video_bgr
