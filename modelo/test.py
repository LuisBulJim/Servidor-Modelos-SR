# flake8: noqa
from basicsr.utils.registry import DATASET_REGISTRY
from custom_dataset import InferenceDataset

DATASET_REGISTRY.register(InferenceDataset)

# Mantener el resto del código original de BasicSR
import modelo.archs
import modelo.data
import modelo.models


# Importaciones estándar de Python
import os
import sys
import logging
import os.path as osp

# Bibliotecas de PyTorch
import torch

# Funciones y utilidades de basicsr
import basicsr
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info,get_root_logger,get_time_str,make_exp_dirs
from basicsr.utils.options import dict2str,parse_options


if __name__ == '__main__':
    # Antes de test_pipeline(root_path) en modelo/test.py
    from basicsr.data import build_dataset

    # Parchear la clase Dataset
    original_build_dataset = build_dataset

    def patched_build_dataset(opt):
        dataset = original_build_dataset(opt)
        if not hasattr(dataset, 'opt'):
            dataset.opt = opt  # Inyectar el atributo opt
        return dataset

    # Aplicar el parche
    import basicsr.test
    basicsr.test.build_dataset = patched_build_dataset
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    
    def upscaler(config_path):
        # Parsear las opciones de configuración
        opt, _ = parse_options(config_path, is_train=False) 
            #is_train=False indica que el modelo se utiliza en modo de prueba, no para entrenamiento.
            #opt contiene las configuraciones del modelo, datos, y rutas necesarias para la ejecución.

        # Configura PyTorch para rendimiento
        torch.backends.cudnn.benchmark = True

        # Crea los directorios necesarios para el experimento.
        make_exp_dirs(opt)
        #log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
        #logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
            #activar para generar el archivo logger

        # Crea los datasets y dataloaders
        test_loaders = []
        for _, dataset_opt in sorted(opt['datasets'].items()):
            test_set = build_dataset(dataset_opt)  # Crea un conjunto de datos utilizando las opciones del archivo de configuración.
            test_loader = build_dataloader(        # Construye un dataloader, que maneja el procesamiento por lotes y la paralelización.
                test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            test_loaders.append(test_loader)

        # Crea modelo
        model = build_model(opt)

        for test_loader in test_loaders:
            test_set_name = test_loader.dataset.opt['name']
            print(f'Mejorando la resolución ...')
            model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
            # Llama al método de validación para aplicarSR a cada imagen del dataloader
            print(f'Mejora finalizada.')
    
    
    
    upscaler(root_path)
