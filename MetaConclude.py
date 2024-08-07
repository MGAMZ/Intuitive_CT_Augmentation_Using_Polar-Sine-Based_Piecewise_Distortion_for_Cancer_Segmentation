import os
import pdb
import json
from os import path as osp
from tqdm import tqdm
from typing import Hashable, Dict, SupportsInt, List, Tuple, Iterable
from collections import defaultdict, Counter

import pydicom
import numpy as np
import pandas as pd



def fetch_one_dcm_for_each_scan(dcm_root:str):
    for root, dirs, files in tqdm(os.walk(dcm_root), desc='Processing'):
        for file in files:
            if file.endswith('.dcm'):
                yield os.path.join(root, file)
                break

def scan_dicoms(dcm_root, save_path):
    dcm_iterator = fetch_one_dcm_for_each_scan(dcm_root)
    
    temp = {
        'dates': [],
        'software_version': [],
        'slice_thickness': [],
        'protocol': [],
        'filter': [],
    }
    for dcm_path in dcm_iterator:
        ds = pydicom.dcmread(dcm_path)
        temp['dates'].append(int(ds.get('InstanceCreationDate')))
        temp['software_version'].append(ds.get('SoftwareVersions'))
        temp['slice_thickness'].append(ds.get('SliceThickness'))
        temp['protocol'].append(ds.get('ProtocolName'))
        temp['filter'].append(ds.get('FilterType'))

    pf = pd.DataFrame(temp)
    pf.to_csv(save_path + '.csv', index=False)



if __name__ == '__main__':
    dcm_root = '/mnt/e/mgam_CT/2023_Med_CQK/'
    dicom_metas = './dcm_metas'
    scan_dicoms(dcm_root, dicom_metas)
