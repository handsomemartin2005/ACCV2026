import torch, timm
import os
from thop import clever_format, profile

# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(timm.list_modules())
print(timm.list_models())
input = torch.randn([1, 3, 224, 224]).to(device)

# features_only 是一个用于控制是否输出中间指令的参数
# swin_base_patch4_window7_224
# efficientvit_b0
model = timm.create_model('resnet50',
                          pretrained=False,
                          # img_size=640,
                          features_only=True)
model.to(device)
model.eval()

# print(model.feature_info.channels())
# for feature in model(input):
#     print(feature.size())

macs, params = profile(model.to(device), (input,), verbose=False)
flops, params = clever_format([macs*2, params], "%.3f") # 计算MACs -> Flops = 2 * MACs

print("Total_FLOPs: %s" % (flops))
print("Total_params: %s" % (params))
