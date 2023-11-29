import sys
sys.path.append("..")

import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from rdkit import Chem
from scipy import sparse as sp
import argparse 

from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

def calculate_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512))

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--data_path", type=str, default='../datasets')
    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=32)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    with open(f"{args.data_path}/pubchem-10m-clean.txt", 'r') as f:
            lines = f.readlines()
            smiless = [line.strip('\n') for line in lines]

    print('extracting fingerprints')
    pool = Pool(processes=128)
    FP_list = list(tqdm(pool.imap(calculate_fingerprint, smiless), total=len(smiless)))
    pool.close()
    pool.join()

    print('process complete')
    FP_arr = np.array(FP_list)
    print('convert to numpy array success')
    del FP_list
    print('delete FP_list success')
    FP_sp_mat = sp.csc_matrix(FP_arr)
    print('saving fingerprints')
    sp.save_npz(f"{args.data_path}/rdkfp1-7_512.npz", FP_sp_mat)

    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    pool = Pool(processes=128)  # use 128 processes
    features_map = Pool(args.n_jobs).imap(generator.process, tqdm(smiless))
    pool.close()
    pool.join()

    arr = np.array(list(features_map))
    del features_map
    print('saving descriptors')
    np.savez_compressed(f"{args.data_path}/molecular_descriptors.npz",md=arr[:,1:])