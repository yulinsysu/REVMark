import torch
import torch.nn as nn
from MotionEstimation import ME_Spynet, flow_warp

class TAsBlock(nn.Module):
    def __init__(self, framenum, enable=False):
        super(TAsBlock, self).__init__()
        self.framenum = framenum
        self.enable = enable
        self.motion_estimate = ME_Spynet(levelnum=4)
        self.motion_estimate.load('ME_Spynet_Full.pth')
        self.motion_estimate.eval()
        self.warp = flow_warp
        self.block = nn.Sequential(
            nn.Conv3d(3, 32, 3, stride=1, padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 3, 3, stride=1, padding=1), nn.LeakyReLU(inplace=True)
        )

    def flow_split(self, video): # video:(B,C,F,H,W)
        ref_idx = 4
        frameref = video[:,:,ref_idx]
        vidrec = []
        flow = []
        for f in range(self.framenum):
            if f == ref_idx:
                vidrec.append(frameref)
                flow.append(None)
                continue
            mv = self.motion_estimate(frameref/2+0.5, video[:,:,f]/2+0.5).detach()
            warped = self.warp(video[:,:,f], mv, self.motion_estimate.coords)
            vidrec.append(warped)
            flow.append(mv)
        vidrec = torch.stack(vidrec, dim=2)
        return vidrec, flow

    def flow_warp_repair(self, frame, flow, coords):
        _, _, H, W = frame.shape
        flow = coords[str(list(flow.shape[2:4]))]*0.99 + torch.cat([flow[:,0:1]/((W-1)/2), flow[:,1:2]/((H-1)/2) ], dim=1)
        return torch.nn.functional.grid_sample(input=frame, grid=flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)

    def flow_merge(self, video, flow): # video:(B,C,F,H,W)
        ref_idx = 4
        vidrec = []
        for f in range(self.framenum):
            if f == ref_idx:
                vidrec.append(video[:,:,f])
                continue
            warped = self.flow_warp_repair(video[:,:,f], -flow[f], self.motion_estimate.coords)
            vidrec.append(warped)
        vidrec = torch.stack(vidrec, dim=2)
        return vidrec
    
    def forward(self, video): # video:(B,C,F,H,W)
        if not self.enable:
            return video
        x, flow = self.flow_split(video)
        x = self.block(x)
        x = self.flow_merge(x, flow)
        return x
