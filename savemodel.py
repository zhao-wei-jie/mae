import torch
import os.path as osp
pth='output_dir/20220418_1019/checkpoint-799.pth'
basedir=osp.dirname(pth)
print(basedir)
ckpoint=torch.load(pth)
# print(ckpoint['model'])
save_ck = {}
del ckpoint['model']['decoder_pred.weight']
del ckpoint['model']['decoder_pred.bias']
save_ck['model'] = ckpoint['model']
print(basedir+'/'+osp.basename(pth).replace('checkpoint','model'))
torch.save(save_ck,basedir+'/'+osp.basename(pth).replace('checkpoint','model'))