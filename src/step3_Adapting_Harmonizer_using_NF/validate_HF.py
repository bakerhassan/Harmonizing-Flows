# %%

from torch.autograd import Variable
import nibabel as nib
## Standard libraries
import os
from src.globals import globals
from pathlib import Path
from nilearn.image import resample_img

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# DataLoader
from src.step3_Adapting_Harmonizer_using_NF.hf_dataloader import MedicalImage2DDataset
from src.step3_Adapting_Harmonizer_using_NF.NF_model import flow_model
from src.step3_Adapting_Harmonizer_using_NF.HF_model import FLow_harmonizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

source_site = 'CALTECH'
target_site = 'healthy'

print(f'source:{source_site}  target:{target_site}')

net_harmonizer = torch.load(f'../checkpoints/UNet2D_harmonizer/model/Best_UNet2D_harmonizer.pkl').cuda()
flow = flow_model(dequant_mode='variational').cuda()

print("Found pretrained model, loading...")
# TODO this is a dirty fix ...
ckpt = torch.load(
    f"../checkpoints/ABIDE-FLOW-{source_site}/ABIDE-Guided-Flow-variational/lightning_logs/version_18503795/checkpoints/last.ckpt",
    map_location=globals.device)
flow.load_state_dict(ckpt['state_dict'])
flow.eval()
print('flow model loaded')

model = FLow_harmonizer(flow, net_harmonizer)
for param in model.flows.parameters():
    param.requires_grad = False
model.flows.eval()
root_dir = '../../data/'

optimizer = optim.Adam(model.harmonizer.parameters(), lr=4e-6)
model.zero_grad()
optimizer.zero_grad()
for file in os.listdir(globals.target_data):
    file = os.path.join(globals.target_data, file)
    if os.path.isfile(file):
        if not file.endswith('.nii') and not file.endswith('.nii.gz') and (
                file.endswith('seg.nii.gz') or file.endswith('seg.nii')):
            continue
        print(file)
        test_set = MedicalImage2DDataset(globals.affine_file, file)
        test_loader = DataLoader(test_set, batch_size=512, num_workers=4, shuffle=False)
        for i in range(20):
            bpds = []
            for batch_idx, data in enumerate(test_loader):
                img = data
                bpd = model(Variable(img).cuda())
                loss = bpd.mean()
                bpds.append(bpd.data.detach().cpu().numpy())
                loss.backward()
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()
        harmonized_data = []
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                harmonized_data.extend(net_harmonizer(data))
        harmonized_data = torch.concat(harmonized_data)
        harmonized_data = harmonized_data.cpu().numpy()
        affine, _ = test_set.get_info()
        image = nib.Nifti1Image(harmonized_data, affine)
        nib.save(image, globals.harmonized_results_path + file.split("/")[-1])
