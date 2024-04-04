import math
import inspect
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class AttnMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin_layers = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd, bias=True) for _ in range(config.n_linear)])
        self.dropout = nn.Dropout(config.dropout)
        self.ln_layers = nn.ModuleList([LayerNorm(config.n_embd, bias=config.bias) for _ in range(config.n_linear)])

    def forward(self, x):
        for linear, ln in zip(self.lin_layers, self.ln_layers):
            x = new_gelu(linear(ln(x)))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # proj function
        self.value_mlp = AttnMLP(config)
        self.query_and_key_mlp = AttnMLP(config)
        # Q, K, V transformation
        self.query_transform = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.key_transform = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value_transform = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # support only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() 
        unshare_x = self.value_mlp(x)
        share_x = self.query_and_key_mlp(x)
        q = self.query_transform(share_x)
        k = self.key_transform(share_x)
        v = self.value_transform(unshare_x)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class COLAConfig:
    seed: int = 0
    data: str = ''
    datapath: str = ''
    min_seq_len: int = 6
    max_seq_len: int = 24
    use_start_letter: bool = True
    start_letter: int = 0
    device: torch.device = torch.device('cuda:0')
    block_size: int = 24
    vocab_size: int = 20000 
    token_size: int = 20001
    n_linear: int = 1
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 96
    dropout: float = 0.0
    bias: bool = True 
    
    meta_lr: float = 1e-3
    update_lr: float = 0.01
    meta_epochs: int= 5
    city_epochs: int = 1
    test_epochs: int = 50
    domain_specific_params: list = field(default_factory=lambda:['value_mlp', 'value_transform', 'wte'])

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

class COLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        if self.config.use_start_letter:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.token_size, config.n_embd, padding_idx=config.token_size-1), 
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))
            self.lm_head = nn.Linear(config.n_embd, config.token_size, bias=False)
        else:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd), 
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 
        
        # !!! ATTENTION !!!
        # duplicate parameters will not be included in 'named_parameters', i.e., the parameters of lm_head are private as same as wte and not involved in meta updating.
        # if changing the order of initialization, SPEC_DICT['sharemlp'] should add 'lm_head' to ensure the wte parameters (copy from lm_head) are not shared by all cities.
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def copy_invariant_params(self, city_model):
        for (m_name, m_param), (c_name, c_param) in zip(self.named_parameters(), city_model.named_parameters()):
            # !!! ATTENTION !!!
            contains_specific = any(sub_str in m_name for sub_str in self.config.domain_specific_params)
            if not contains_specific:
                assert m_name == c_name
                c_param.data = m_param.data.clone()
                assert torch.allclose(c_param.data, m_param.data)

    def get_acc_topk(self, preds, targets):
        acc_K = [1, 5, 10, 20]
        result = {}
        totalMRR = []
        for K in acc_K:
            result[K] = 0

        seq_len_l = []
        for i in range(len(preds)):  
            max_len = self.config.max_seq_len if self.config.use_start_letter else self.config.max_seq_len - 1
            seq_len = max_len - len(torch.where(targets[i]==-1)[0])
            seq_len_l.append(seq_len)

            for j in range(seq_len): 
                pred, target = preds[i][j], targets[i][j].item()
                sortedPred = torch.topk(pred, len(pred))[1].tolist()
                truthIndex = sortedPred.index(target) + 1
                avgPrec = 1 / truthIndex
                totalMRR.append(avgPrec)

                sorted_indexs = {}
                for K in acc_K:  
                    sorted_indexs[K] = sortedPred[:K]
                    if target in sorted_indexs[K]:
                        result[K] += 1
        
        result['num_of_test'] = sum(seq_len_l)
        result['mrr'] = np.sum(totalMRR)
        result['mrr_num'] = len(totalMRR)
        return result

    def forward(self, idx, targets=None, freqs=None, use_acc = False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) 
        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            if self.config.use_start_letter:
                logits[:, :, -1] = float('-inf')
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]) 
            if self.config.use_start_letter:
                logits[:, :, -1] = float('-inf')
            loss = None
        if use_acc:
            accs = self.get_acc_topk(logits, targets)
            return logits, loss, accs
        else:
            return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn 
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        decay.remove('lm_head.weight')
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12 
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, args, len_list, num_samples, temperature=1.0, top_k=None): 
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
   
        args.freqs = args.freqs / args.freqs.sum()
        adjustments = np.log(args.freqs ** args.balance_coef + 1e-12)  
        adjustments = torch.from_numpy(adjustments)
        adjustments = adjustments.to(args.device)

        len_vals, len_cnts = np.unique(len_list, return_counts=True) 
        len_cnts = len_cnts / np.sum(len_cnts)
        samples_len = [np.random.choice(len_vals, 1, p = len_cnts)[0] for i in range(num_samples)]

        if self.config.use_start_letter:
            idx = torch.LongTensor([self.config.start_letter]*num_samples).reshape(-1, 1).to(args.device) 
            gen_seq_len = self.config.max_seq_len
            get_seq_offset = 1
        else:
            start_dist=torch.tensor(np.load(f'{self.config.datapath}/{self.config.data}/start.npy')).float()
            idx = torch.LongTensor([torch.multinomial(start_dist, 1) for _ in range(num_samples)]).reshape(-1, 1).to(args.device) 
            gen_seq_len = self.config.max_seq_len - 1
            get_seq_offset = 0
         
        for t in range(gen_seq_len):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]  
            logits, _ = self(idx_cond)
            # adjust logits when generating
            logits = logits - adjustments
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        pred = []
        for i in range(len(idx)):
            seq = idx[i][get_seq_offset:samples_len[i]+get_seq_offset]
            pred.append(list(seq.cpu().numpy()))
        return pred

    @torch.no_grad()
    def generate_for_seir(self, args, num_samples, temperature=1.0, top_k=None): 
        args.freqs = args.freqs / args.freqs.sum()
        adjustments = np.log(args.freqs ** args.balance_coef + 1e-12)  
        adjustments = torch.from_numpy(adjustments)
        adjustments = adjustments.to(args.device)
        start_dist=torch.tensor(np.load(f'{self.config.datapath}/{self.config.data}/start.npy')).float()
        idx = torch.LongTensor([torch.multinomial(start_dist, 1) for _ in range(num_samples)]).reshape(-1, 1).to(args.device) 
        gen_seq_len = 24*7-1
    
        for t in tqdm(range(gen_seq_len)):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]  
            logits, _ = self(idx_cond)
            # adjust logits when generating
            logits = logits - adjustments
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        pred = []
        for i in range(len(idx)):
            seq = idx[i][:]
            pred.append(list(seq.cpu().numpy()))
        return pred