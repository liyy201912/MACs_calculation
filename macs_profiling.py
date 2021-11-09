import torch
import torch.nn as nn
import torchvision


class ProfileConv(nn.Module):
    def __init__(self, model):
        super(ProfileConv, self).__init__()
        self.model = model
        self.hooks = []
        self.macs = []

        def hook_conv(module, input, output):
            self.macs.append(output.size(1) * output.size(2) * output.size(3) *
                             module.weight.size(-1) * module.weight.size(-1) * input[0].size(1) / module.groups)

        def hook_linear(module, input, output):
            self.macs.append(module.weight.size(0) * module.weight.size(1))

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.hooks.append(module.register_forward_hook(hook_conv))
            elif isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(hook_linear))

    def forward(self, x):
        self.model.to(x.device)
        _ = self.model(x)
        for handle in self.hooks:
            handle.remove()
        return self.macs


if __name__ == '__main__':

    # find the 'out = model(x)' in your code, my method is based on pytorch hook
    # example input
    x = torch.randn(1, 3, 224, 224)
    # example model
    model = torchvision.models.mobilenet_v2(pretrained=False)

    profile = ProfileConv(model)
    MACs = profile(x)

    print('number of conv&fc layers:', len(MACs))
    print(sum(MACs) / 1e9, 'GMACs, only consider conv layers')
