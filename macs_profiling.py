import torch
import torch.nn as nn
import torchvision


class ProfileConv(nn.Module):
    def __init__(self, model):
        super(ProfileConv, self).__init__()
        self.model = model
        self.hooks = []
        self.macs = []
        self.params = []

        def hook_conv(module, input, output):
            self.macs.append(output.size(1) * output.size(2) * output.size(3) *
                             module.weight.size(-1) * module.weight.size(-1) * input[0].size(1) / module.groups)
            self.params.append(module.weight.size(0) * module.weight.size(1) *
                               module.weight.size(2) * module.weight.size(3) + module.weight.size(1))

        def hook_linear(module, input, output):
            if len(input[0].size()) > 2:
                self.macs.append(module.weight.size(0) * module.weight.size(1) * input[0].size(-2))
            else:
                self.macs.append(module.weight.size(0) * module.weight.size(1))
            self.params.append(module.weight.size(0) * module.weight.size(1) + module.bias.size(0))

        def hook_gelu(module, input, output):
            if len(output[0].size()) > 3:
                self.macs.append(output.size(1) * output.size(2) * output.size(3))
            else:
                self.macs.append(output.size(1) * output.size(2))

        def hook_layernorm(module, input, output):
            self.macs.append(2 * input[0].size(1) * input[0].size(2))
            self.params.append(module.weight.size(0) + module.bias.size(0))

        def hook_avgpool(module, input, output):
            self.macs.append(output.size(1) * output.size(2) * output.size(3) * module.kernel_size * module.kernel_size)

        def hook_attention(module, input, output):
            self.macs.append(module.key_dim * (module.resolution ** 4) * module.num_heads +
                             module.dh * (module.resolution ** 4))

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.hooks.append(module.register_forward_hook(hook_conv))
            elif isinstance(module, nn.Linear):
                self.hooks.append(module.register_forward_hook(hook_linear))
            elif isinstance(module, nn.GELU):
                self.hooks.append(module.register_forward_hook(hook_gelu))
            elif isinstance(module, nn.LayerNorm):
                self.hooks.append(module.register_forward_hook(hook_layernorm))
            elif isinstance(module, nn.AvgPool2d):
                self.hooks.append(module.register_forward_hook(hook_avgpool))
#             elif isinstance(module, Attention):
#                 self.hooks.append(module.register_forward_hook(hook_attention))

    def forward(self, x):
        self.model.to(x.device)
        _ = self.model(x)
        for handle in self.hooks:
            handle.remove()
        return self.macs, self.params


if __name__ == '__main__':

    # find the 'out = model(x)' in your code, my method is based on pytorch hook
    # example input
    x = torch.randn(1, 3, 224, 224)
    # example model
    model = torchvision.models.mobilenet_v2(pretrained=False)

    profile = ProfileConv(model)
    MACs, params = profile(x)

    print('number of conv&fc layers:', len(MACs))
    print(sum(MACs) / 1e9, 'GMACs')
    print(sum(params) / 1e6, 'M parameters')
