import torch
import torch.nn as nn
# reference: Optical Flow Estimation Using a Spatial Pyramid Network, Ranjan, Anurag and Black, Michael J, CVPR 2017

def flow_warp(frame, flow, coords):
    _, _, H, W = frame.shape
    flow = coords[str(list(flow.shape[2:4]))] + torch.cat([flow[:,0:1]/((W-1)/2), flow[:,1:2]/((H-1)/2) ], dim=1)
    return torch.nn.functional.grid_sample(input=frame, grid=flow.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')

class MEBasic(nn.Module):
    def __init__(self):
        super(MEBasic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(8, 32, 7, 1, padding=3), nn.ReLU(),
            nn.Conv2d(32, 64, 7, 1, padding=3), nn.ReLU(),
            nn.Conv2d(64, 32, 7, 1, padding=3), nn.ReLU(),
            nn.Conv2d(32, 16, 7, 1, padding=3), nn.ReLU(),
            nn.Conv2d(16, 2, 7, 1, padding=3)
        )
    def forward(self, x):
        return self.model(x)

class ME_Spynet(nn.Module): # get flow
    def __init__(self, levelnum=4):
        super().__init__()
        self.levelnum = levelnum
        self.moduleBasic = torch.nn.ModuleList([ MEBasic() for i in range(levelnum) ])
        self.coords = {}
    
    def load(self, state_dict_path):
        state_dict = torch.load(state_dict_path)
        for k in dict(state_dict):
            if int(k.split('.')[1]) >= self.levelnum:
                del state_dict[k]
        self.load_state_dict(state_dict)

    def forward(self, im1, im2):
        im1list = [im1]
        im2list = [im2]
        for L in range(self.levelnum-1):
            im1list.insert(0, torch.nn.functional.avg_pool2d(im1list[0], kernel_size=2, stride=2))
            im2list.insert(0, torch.nn.functional.avg_pool2d(im2list[0], kernel_size=2, stride=2))

        init_shape = [im1.shape[0], 2, im1list[0].shape[2] // 2, im1list[0].shape[3] // 2]
        flowfileds = torch.zeros(init_shape, dtype=torch.float32, device=im1.device)
        for L in range(self.levelnum):
            flowfiledsUpsample = 2.0 * torch.nn.functional.interpolate(flowfileds, scale_factor=2, mode='bilinear', align_corners=False)
            
            _, _, H, W = flowfiledsUpsample.shape
            if str([H,W]) not in self.coords: # make absolute coordinate
                coordx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(-1, -1, H, -1)
                coordy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(-1, -1, -1, W)
                self.coords[str([H,W])] = torch.cat([coordx,coordy], dim=1).to(im1.device)
            
            im2_warp = flow_warp(im2list[L], flowfiledsUpsample, self.coords)
            flowfileds = flowfiledsUpsample + self.moduleBasic[L](torch.cat([im1list[L], im2_warp, flowfiledsUpsample], dim=1)) # residualflow

        return flowfileds
