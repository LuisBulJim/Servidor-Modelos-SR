# modelo/custom_dataset.py
import os
import torch
from torch.utils.data import Dataset
from basicsr.utils import scandir, imfrombytes, img2tensor

class InferenceDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt  # Guardar la configuración manualmente
        self.lq_folder = opt['dataroot_lq']
        self.paths = sorted([os.path.join(self.lq_folder, x) for x in scandir(self.lq_folder)])
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        self.color = opt.get('color', 'rgb')

    def __getitem__(self, index):
        lq_path = self.paths[index]
        with open(lq_path, 'rb') as f:
            img_bytes = f.read()
        img_lq = imfrombytes(img_bytes, float32=True)
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        return {'lq': img_lq, 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)