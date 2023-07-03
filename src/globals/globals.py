import os
import torch

HPC = os.getenv('SLURM_CPUS_ON_NODE') != None
if HPC:
    print("I'm on HPC!")
    lustre_path = '/lustre/scratch/bakerh/'
    training_data_location = lustre_path + '/Caltech/derivative/normal/training/'
    validation_data_location = lustre_path + '/Caltech/derivative/normal/validation/'
    affine_file = lustre_path + 'affine.npy'
    harmonized_results_path = lustre_path + 'harmonized_data/'
    target_data = lustre_path + 'ds003097-derivative/'
else:
    lustre_path = '/Users/hassan/Downloads/Caltech/derivative'
    training_data_location = '/Users/hassan/Downloads/Caltech/derivative/normal/training'
    validation_data_location = '/Users/hassan/Downloads/Caltech/derivative/normal/validation/'
    affine_file = '../../affine.npy'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
slice_size = (224, 224)