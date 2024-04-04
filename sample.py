import os
import torch
from cola import COLAConfig, COLA
import helpers
import argparse
import socket

SPEC_DICT = {
    'sharemlp': ['value_mlp', 'value_transform', 'wte']
}
CITY_SIM={
    'New_York': 'NYC',
    # 'Tokyo': 'TKY',
    # 'Bangkok': 'BGK',
    'geolife': 'GEO',
    'yahoo_japan': 'JPN',
    'Singapore': 'SGP'
}
parser = argparse.ArgumentParser()
parser.add_argument('--seed',  default=0, type=int, choices=[0,1,2,3,4])   
parser.add_argument('--cuda',  default="0", type=str)
parser.add_argument('--dtype', default='float16', type=str)
parser.add_argument('--method', default='Cross-city-Mobility-Transformer', type=str)
parser.add_argument('--spec_type', default='sharemlp', type=str) 
parser.add_argument('--domain_specific_params', nargs='+', default=['value_mlp', 'value_transform', 'wte']) 
parser.add_argument('--balance_coef', default=0.1, type=float)
parser.add_argument('--load_type', default='last', choices=['best', 'last'])
parser.add_argument('--train_cities', nargs='+', default=['geolife', 'yahoo_japan', 'Singapore'])
parser.add_argument('--data', type=str, default='New_York')
parser.add_argument('--min_seq_len', default='6', type=int) 
parser.add_argument('--top_k',  default=200, type=int)   
parser.add_argument('--num_samples', default=5000, type=int)
parser.add_argument('--use_start_letter', action='store_true')
parser.add_argument('--start_letter', default=0, type=int) 
parser.add_argument('--out_dir',  default="out", type=str)   
parser.add_argument('--init_from',  default="resume", type=str)   
parser.add_argument('--temperature',  default="0.8", type=float)   

# setting for meta model
parser.add_argument('--meta_epochs', default=5, type=int)
parser.add_argument('--city_epochs', default=1, type=int)
parser.add_argument('--test_epochs', default=50, type=int)
parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=5e-4)
parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-3)
parser.add_argument('--dis_type',  default="geo", type=str, choices=['euclidean', 'geo'])   
parser.add_argument('--n_linear', default=1, type=int)  

args = parser.parse_args()
args.domain_specific_params = SPEC_DICT[args.spec_type]

helpers.set_random_seed(args.seed)
args.hostname = socket.gethostname()
args.datapath = f'./rawData/Foursquare_global/city_user_day_fix/min_seq_{args.min_seq_len}'
args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else torch.device("cpu"))
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
args.ctx = torch.autocast(device_type='cuda', dtype=ptdtype)

data_neural, _ = helpers.read_meta_datasets(test_city=args.data, path=args.datapath, dis_type=args.dis_type)
args.vocab_size = data_neural[args.data]['num_locs']
args.max_dist = data_neural[args.data]['max_dist']
args.freqs = data_neural[args.data]['frequency']

args.out_dir = args.out_dir + '/main' 
postfix = f'me{args.meta_epochs}-ce{args.city_epochs}-te{args.test_epochs}-mlr{args.meta_lr}-tlr{args.update_lr}'
train_str = f'{args.seed}-{args.spec_type}-ln{args.n_linear}-{postfix}'
post_str = '-post'
test_str = f'{args.seed}-{args.spec_type}{post_str}-{args.balance_coef}-{args.load_type}-ln{args.n_linear}-{postfix}'

args.out_dir = f'{args.out_dir}/ln{args.n_linear}'
path_test_model = f'{args.out_dir}/{args.data}_{train_str}_ckpt.pth'
path_test_model_last = f'{args.out_dir}/{args.data}_{train_str}_ckpt_last.pth'
path_test_gene = f'{args.out_dir}/{args.data}_{test_str}_gene.txt'

log_dir = f'./logs'
log_prefix=f'{args.method}-{args.data}-{train_str}-sample-{args.hostname}-gpu{args.cuda}'
logger = helpers.set_logger(log_dir=log_dir, log_prefix=log_prefix)
logger.info(args)

# load model
if args.init_from == 'resume':
    if args.load_type == 'best':
        logger.info(f'load test model from {path_test_model}')
        ckpt = torch.load(path_test_model, map_location=args.device)
    else:
        logger.info(f'load test model from {path_test_model_last}')
        ckpt = torch.load(path_test_model_last, map_location=args.device)
    colaconf = COLAConfig(**ckpt['model_args'])
   
    model = COLA(colaconf)
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
else:
    logger.info("wrong assignment!")

model.eval()
model.to(args.device)

# load test data
len_list = data_neural[args.data]['tra_len']
data_traj = data_neural[args.data]['tes_path']

# generate trajectories
with torch.no_grad():
    with args.ctx:
        pred_data = model.generate(args, len_list, args.num_samples, temperature=args.temperature, top_k=args.top_k)
        with open(path_test_gene, 'w') as f:
            for path in pred_data:
                f.write(' '.join([str(poi) for poi in path]))
                f.write('\n')