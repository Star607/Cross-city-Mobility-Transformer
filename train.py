import argparse
import os
from math import ceil
from pathlib import Path
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import helpers
from cola import COLA, COLAConfig 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

SPEC_DICT = {
    # !!! ATTENTION !!!
    # if changing the order of initialization of wte and lm_head in COLA, there should add 'lm_head' to ensure the wte parameters (copy from lm_head) are not shared by all cities.
    'sharemlp': ['value_mlp', 'value_transform', 'wte']
}
CITY_SIM={
    'New_York': 'NYC',
    'geolife': 'GEO',
    'yahoo_japan': 'JPN',
    # 'Tokyo': 'TKY',
    # 'Bangkok': 'BGK',
    'Singapore': 'SGP'
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',  default=0, type=int, choices=[0,1,2,3,4])   
    parser.add_argument('--cuda',  default="0", type=str)   
    parser.add_argument('--dtype', default='float16', type=str)
    parser.add_argument('--method', default='Cross-city-Mobility-Transformer', type=str) 
    parser.add_argument('--train_cities', nargs='+', default=['geolife', 'yahoo_japan', 'Singapore'])
    parser.add_argument('--data', type=str, default='New_York')
    parser.add_argument('--spec_type', default='sharemlp', type=str) 
    parser.add_argument('--domain_specific_params', nargs='+', default=['value_mlp', 'value_transform', 'wte']) 
    parser.add_argument('--datapath', default='', type=str)
    parser.add_argument('--out_dir',  default="out", type=str)   
    parser.add_argument('--min_seq_len', default='6', type=int) 
    parser.add_argument('--max_seq_len', default='24', type=int) 

    # setting for meta
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=5e-4)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-3)
    parser.add_argument('--meta_epochs', default=5, type=int)
    parser.add_argument('--city_epochs', default=1, type=int)
    parser.add_argument('--test_epochs', default=50, type=int)

    # setting for model
    parser.add_argument('--use_start_letter', action='store_true')
    parser.add_argument('--start_letter', default='0', type=int) 
    parser.add_argument('--batch_size', default='32', type=int) 
    parser.add_argument('--block_size', default='24', type=int) 
    parser.add_argument('--n_head_t', default='2', type=int)  
    parser.add_argument('--n_layer_t', default='2', type=int) 
    parser.add_argument('--n_linear', default=1, type=int) # the number of attn linear layer 
    parser.add_argument('--n_embd', default='96', type=int) 
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--bias', default=False, type=bool)

    # setting for optimization
    parser.add_argument('--grad_clip', default=1.0, type=float, help='clip gradients at this value, or disable if == 0.0')    
    parser.add_argument('--eval_only', default=False, type=bool)
    parser.add_argument('--eval_interval', default='2', type=int)

    args = parser.parse_args()
    args.domain_specific_params = SPEC_DICT[args.spec_type]

    helpers.set_random_seed(args.seed)
    args.hostname = socket.gethostname()
    args.datapath = f'./rawData/Foursquare_global/city_user_day_fix/min_seq_{args.min_seq_len}'

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else torch.device("cpu"))
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    args.ctx = torch.autocast(device_type='cuda', dtype=ptdtype)
    args.out_dir = args.out_dir + '/main' 

    train_str = f'{args.seed}-{args.spec_type}-ln{args.n_linear}-me{args.meta_epochs}-ce{args.city_epochs}-te{args.test_epochs}-mlr{args.meta_lr}-tlr{args.update_lr}'
    
    # set log path
    log_dir = f'./logs'
    log_prefix=f'{args.method}-{args.data}-{train_str}-train-{args.hostname}-gpu{args.cuda}'
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = helpers.set_logger(log_dir=log_dir, log_prefix=log_prefix)
    logger.info(args)

    # set saved path
    args.out_dir = f'{args.out_dir}/ln{args.n_linear}'
    os.makedirs(args.out_dir, exist_ok=True)
    path_meta_model = f'{args.out_dir}/{args.data}_{train_str}_meta.pth'
    path_test_model = f'{args.out_dir}/{args.data}_{train_str}_ckpt.pth'
    path_test_model_last = f'{args.out_dir}/{args.data}_{train_str}_ckpt_last.pth'

    # load data
    data_neural, meta_loc = helpers.read_meta_datasets(args.train_cities, args.data, args.datapath)

    # initialize meta_model
    meta_model_args = dict(seed=args.seed, data="", datapath="", domain_specific_params=args.domain_specific_params, n_linear=args.n_linear, min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len, use_start_letter=args.use_start_letter, start_letter=args.start_letter, device=device, n_layer=args.n_layer_t, n_head=args.n_head_t, n_embd=args.n_embd, block_size=args.block_size, bias=args.bias, vocab_size=meta_loc, token_size=meta_loc+1, dropout=args.dropout, meta_lr=args.meta_lr, update_lr=args.update_lr, meta_epochs=args.meta_epochs, city_epochs=args.city_epochs, test_epochs=args.test_epochs)
    meta_model = COLA(COLAConfig(**meta_model_args)).to(device)
    
    # initialize models and optimizers for meta_train and meta_test 
    model_dict = {}
    optim_dict = {}
    for i in range(len(args.train_cities)+1):
        if i != len(args.train_cities):
            data = args.train_cities[i]
        else:
            data = args.data
        # set COLAConfig
        model_args = dict(seed=args.seed, data=data, datapath=args.datapath, domain_specific_params=args.domain_specific_params, n_linear=args.n_linear, min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len, use_start_letter=args.use_start_letter, start_letter=args.start_letter, device=device, n_layer=args.n_layer_t, n_head=args.n_head_t, n_embd=args.n_embd, block_size=args.block_size, bias=args.bias, vocab_size=data_neural[data]['num_locs'], token_size=data_neural[data]['num_locs']+1, dropout=args.dropout, meta_lr=args.meta_lr, update_lr=args.update_lr, meta_epochs=args.meta_epochs, city_epochs=args.city_epochs, test_epochs=args.test_epochs)
        model = COLA(COLAConfig(**model_args)).to(device)
        model_dict[data] = model
        optim_dict[data] = optim.Adam(model.parameters(), lr=model.config.update_lr)
    
    best_val_acc_top5 = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    for m_epoch in range(args.meta_epochs):
        for c in args.train_cities:
            # copy parameters from meta_model
            meta_model.copy_invariant_params(model_dict[c])
            model_dict[c].train()
            
            # train source cities
            tra_path = data_neural[c]['tra_path'] 
            batch = int(np.ceil(len(tra_path) / args.batch_size))
            for c_epoch in range(args.city_epochs):
                start_i = 0
                for bt in range(batch):
                    end_i = min(start_i + args.batch_size, len(tra_path))
                    if start_i >= end_i:
                        break
                    x_seq, y_seq = helpers.get_batch_city('tra', args, data_neural[c], start_i, end_i, device)
                    with args.ctx:
                        optim_dict[c].zero_grad(set_to_none=True)
                        logits, loss = model_dict[c](x_seq, y_seq, data_neural[c]['frequency'], use_acc = False)
                        loss.backward()
                        optim_dict[c].step()
                    start_i = start_i + args.batch_size

            # evaluate gradient from the test data of the source city
            start_i, end_i = 0, len(data_neural[c]['tes_path'])
            x_seq, y_seq = helpers.get_batch_city('tes', args, data_neural[c], start_i, end_i, device)
            with args.ctx:
                logits, loss = model_dict[c](x_seq, y_seq, data_neural[c]['frequency'], use_acc = False)
            logger.info(f"MetaEpoch: {m_epoch}, Name: {c}, TrajNum: {len(data_neural[c]['tes_path'])}, Loss: {loss.item():.3f}")

            # update meta_model based on the gradient
            meta_model.eval()
            for name, param in model_dict[c].named_parameters():
                contains_specific = any(sub_str in name for sub_str in meta_model.config.domain_specific_params)
                if contains_specific:
                    continue
                param.data -= args.meta_lr * param.grad
    
        # copy parameters from meta_model
        meta_model.copy_invariant_params(model_dict[args.data])
        
        # load training set of the target city
        tra_path = data_neural[args.data]['tra_path']
        model_dict[args.data].train()

        for t_epoch in range(args.test_epochs):
            start_i = 0
            # finetune the target model
            for bt in range(batch):
                end_i = min(start_i + args.batch_size, len(tra_path))
                if start_i >= end_i:
                    break
                x_seq, y_seq = helpers.get_batch_city('tra', args, data_neural[args.data], start_i, end_i, device)
                with args.ctx:
                    optim_dict[args.data].zero_grad(set_to_none=True)
                    logits, loss = model_dict[args.data](x_seq, y_seq, data_neural[args.data]['frequency'], use_acc = False)
                scaler.scale(loss).backward()
                if args.grad_clip != 0.0:
                    scaler.unscale_(optim_dict[args.data])
                    torch.nn.utils.clip_grad_norm_(model_dict[args.data].parameters(), args.grad_clip)
                scaler.step(optim_dict[args.data])
                scaler.update()
                start_i = start_i + args.batch_size

            res = helpers.estimate_loss(model_dict[args.data], args, data_neural[args.data], device)
            logger.info(f"step {t_epoch+1}: train loss {res['tra_loss']:.4f}, val loss {res['val_loss']:.4f}, train acc@5 {res['tra_accs'][1]:.4f}, val acc@5 {res['val_accs'][1]:.4f}")

            if res['val_accs'][1] > best_val_acc_top5:  # res['val_accs'][1] -> top5
                best_val_acc_top5 = res['val_accs'][1]          
                if m_epoch > 0:
                    ckpt = {
                        'model': model_dict[args.data].state_dict(),
                        'optimizer': optim_dict[args.data].state_dict(),
                        'model_args': model_args,
                        'm_epoch': m_epoch,
                        't_epoch': t_epoch,
                        'best_val_acc_top5': best_val_acc_top5
                    }
                    logger.info(f"saving checkpoint to {args.out_dir}")
                    torch.save(ckpt, path_test_model)
                    
        model_dict[args.data].eval()
    
    # save model
    torch.save({'state_dict': meta_model.state_dict()}, path_meta_model)
    ckpt_last = {'model': model_dict[args.data].state_dict(),
                'optimizer': optim_dict[args.data].state_dict(),
                'model_args': model_args,
                'm_epoch': m_epoch,
                't_epoch': t_epoch,
                'best_val_acc_top5': res['val_accs'][1]}
    torch.save(ckpt_last, path_test_model_last)

    if os.stat(path_meta_model).st_uid == os.getuid():
        os.system(f'chmod 777 {path_meta_model}')
        os.system(f'chmod 777 {path_test_model_last}')
    if os.stat(args.out_dir).st_uid == os.getuid():
        logger.info('Change the out_dir status to 777 recursively.')
        os.system(f"chmod 777 {args.out_dir} -R")
    if os.stat(log_dir).st_uid == os.getuid():
        logger.info('Change the log status to 777 recursively.')
        os.system(f"chmod 777 {log_dir} -R")

