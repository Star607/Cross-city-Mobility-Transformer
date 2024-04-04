import torch
from torch.autograd import Variable
from math import ceil
import numpy as np
import random
import logging
from datetime import datetime
import os
from pathlib import Path 
import json
import copy

CITY_SIM={
    'New_York': 'NYC',
    'geolife': 'GEO',
    'yahoo_japan': 'JPN',
    'Singapore': 'SGP'
}

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_logger(log_dir='./logs/', log_prefix=''):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s',
        "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler() 
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    ts = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    fh = logging.FileHandler(f'{log_dir}/{log_prefix}-{ts}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def read_meta_datasets(train_cities=[], test_city='', path='', dis_type='geo'):
    params_map = {"Singapore":{'num_locs':11509, 'max_dist':285.42816162109375, 'max_dist_geo':4371.3525390625},
                "New_York":{'num_locs':9387, 'max_dist':298.0574035644531, 'max_dist_geo':4454.46826171875},
                "geolife":{'num_locs':32675, 'max_dist':350.427001953125, 'max_dist_geo':3943.664306640625},
                "yahoo_japan": {'num_locs':16241, 'max_dist_geo':373.7079162597656}}
    data_neural = {c:{} for c in train_cities + [test_city]}
    meta_loc = 0
    for c in train_cities + [test_city]:
        with open(f'{path}/{c}/split.json', 'r') as fr:  
            data = json.load(fr)
        data_neural[c]['tra_user'] = data['tra_user']
        data_neural[c]['tra_path'] = data['tra_path']
        data_neural[c]['tra_tim'] = data['tra_tim']
        data_neural[c]['tra_len'] = [len(t) for t in data['tra_path']]
        data_neural[c]['tes_user'] = data['tes_user']
        data_neural[c]['tes_path'] = data['tes_path']
        data_neural[c]['tes_tim'] = data['tes_tim']
        data_neural[c]['tes_len'] = [len(t) for t in data['tes_path']]
        data_neural[c]['val_user'] = data['val_user']
        data_neural[c]['val_path'] = data['val_path']
        data_neural[c]['val_tim'] = data['val_tim']
        data_neural[c]['val_len'] = [len(t) for t in data['val_path']]
        data_neural[c]['num_locs'] = params_map[c]['num_locs']
        if dis_type == 'geo':
            data_neural[c]['max_dist'] = params_map[c]['max_dist_geo']
        else:
            data_neural[c]['max_dist'] = params_map[c]['max_dist']
        meta_loc = max(meta_loc, data_neural[c]['num_locs'])
        merged_l = []
        for traj in data_neural[c]['tra_path']:
            merged_l.extend(traj)
        unq, cnt = np.unique(merged_l, return_counts=True)
        freqs = np.ones(data_neural[c]['num_locs'])
        freqs[unq] = cnt
        data_neural[c]['frequency'] = freqs
    return data_neural, meta_loc
    
def add_eos_and_pad_seq(seqs, EOS = None, mode = 'no-eos'):
    max_seq = 24
    valid_len = [len(seq) for seq in seqs]
    for i, seq in enumerate(seqs):
        if valid_len[i] < max_seq:
            if mode == 'add-eos':
                seq.append(EOS)
                valid_len[i] += 1
                if valid_len[i] < max_seq:
                    seq.extend([0] * (max_seq - valid_len[i]))
            else:
                seq.extend([0] * (max_seq - valid_len[i]))
        assert len(seq) == max_seq
    return seqs, valid_len

def get_batch_city(split, args, city_neural, start_i, end_i, device):
    if split == 'tra':
        data_len, data_traj, data_tim = city_neural['tra_len'], city_neural['tra_path'], city_neural['tra_tim']
    elif split == 'val':
        data_len, data_traj, data_tim = city_neural['val_len'], city_neural['val_path'], city_neural['val_tim']
    else:
        data_len, data_traj, data_tim = city_neural['tes_len'], city_neural['tes_path'], city_neural['tes_tim']
    
    data_traj = copy.deepcopy(data_traj[start_i:end_i])
    data_traj, data_len = add_eos_and_pad_seq(data_traj)
    data_traj, data_len = torch.tensor(data_traj).to(device), torch.tensor(data_len).to(device)

    if args.use_start_letter:
        x_seq, y_seq = torch.zeros_like(data_traj), torch.zeros_like(data_traj)
        x_seq[:, 0] = args.start_letter
        x_seq[:, 1:] = data_traj[:, :-1]
        y_seq[:, :] = data_traj
        mask = torch.arange(len(x_seq[0]), dtype=torch.float32, device=y_seq.device)[None, :] < data_len[:, None]
    else:
        b, t = data_traj.shape[0], data_traj.shape[1]-1
        x_seq, y_seq = torch.zeros((b, t)).to(data_traj), torch.zeros((b, t)).to(data_traj)
        x_seq[:, :] = data_traj[:, :-1]
        y_seq[:, :] = data_traj[:, 1:]
        mask = torch.arange(len(x_seq[0]), dtype=torch.float32, device=y_seq.device)[None, :] < data_len[:, None] - 1
    y_seq[~mask] = -1 
    return x_seq, y_seq

def partial_metrics_sum(results):
    acc_K = [1, 5, 10, 20]
    result = {}
    num_of_test = np.sum([r['num_of_test'] for r in results])
    for K in acc_K:
        result[K] = np.sum([r[K] for r in results])
        result[K] /= num_of_test
    result['mrr'] = np.sum([r['mrr'] for r in results])
    result['mrr'] /= np.sum([r['mrr_num'] for r in results])
    return result[acc_K[0]], result[acc_K[1]], result[acc_K[2]], result[acc_K[3]], result['mrr']

def estimate_loss(model, args, city_neural, device):
    out = {}
    model.eval()
    for split in ['tra', 'val']:
        start_i = 0
        name = f'{split}_path'
        data_traj = city_neural[name]
        batch = int(np.ceil(len(data_traj) / args.batch_size))

        res_l = []
        losses = torch.zeros(batch)
        for bt in range(batch):
            end_i = min(start_i + args.batch_size, len(data_traj))
            X, Y = get_batch_city(split, args, city_neural, start_i, end_i, device)
            with args.ctx:
                logits, loss, res = model(X, Y, freqs=city_neural['frequency'], use_acc=True)
            losses[bt] = loss.item()
            res_l.append(res)
            start_i = start_i + args.batch_size
        eval_res = partial_metrics_sum(res_l)
        out[split+'_accs'] = eval_res
        out[split+'_loss'] = losses.mean()
    model.train()
    return out

def read_data_from_file(fp):
    path = []
    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pois = line.split(' ')
            path.append([int(poi) for poi in pois])
    return path

def get_gps(gps_file):
    gps = np.load(gps_file)
    X, Y= gps[:,0], gps[:,1]
    return X, Y