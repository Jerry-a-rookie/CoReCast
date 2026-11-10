                                                                         

                                                                             
                                                                      
   
import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from scipy.signal import lfilter, lfilter_zi
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data import data_provider
from core.models import model_manager


class Indexed(Dataset):
    def __init__(self, dataset, origin):
        self.dataset, self.origin = dataset, origin
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        x, y, xmark, _ = self.dataset[i]
        return x, y, xmark, i + self.origin


class Retrieval:
    def __init__(self, all_data, train_end, l, h, args, device, center_values=False):
        self.l, self.h, self.args, self.device = l, h, args, device
        self.center_values = center_values
        a = args.ema_alpha
                                                                                
        z = lfilter_zi([a], [1, -(1-a)])[:, None] * all_data[0][None, :]
        trend, _ = lfilter([a], [1, -(1-a)], all_data, axis=0, zi=z)
        trend = trend.astype(np.float32)
        self.trend = torch.as_tensor(trend, device=device)
        self.season = torch.as_tensor(all_data.astype(np.float32), device=device) - self.trend
        self.starts = torch.arange(0, max(0, train_end-l-h+1), args.candidate_stride, device=device)
        self.no_candidate = 0
        self.queries = 0
        self.channel_fallback = 0
        self.selected_counts = {}
        self.cache = {}
        self.cache_bytes = 0

    @torch.no_grad()
    def component(self, series, q, starts, offset, is_trend):
        l, h, a = self.l, self.h, self.args
        c = series.shape[1]
        fallback = (torch.zeros((h, c), device=self.device, dtype=series.dtype)
                    if self.center_values else
                    (offset.clone() if is_trend else torch.zeros((h, c), device=self.device, dtype=series.dtype)))
        if not starts.numel():
            empty_labels = np.full(c, -1, np.int16)
            empty_starts = np.zeros((0, a.topm), np.int32)
            empty_weights = np.zeros((0, a.topm), np.float32)
            empty_counts = np.zeros((0,), np.uint8)
            return fallback.expand(h, c).clone(), c, (empty_labels, empty_starts, empty_weights, empty_counts)
        query = series[q:q+l].T
        keys = series[starts[:, None] + torch.arange(l, device=self.device)[None, :]].permute(2, 0, 1)
        query = query - query.mean(-1, keepdim=True)
        keys = keys - keys.mean(-1, keepdim=True)
        qnorm = torch.linalg.vector_norm(query, dim=-1, keepdim=True)
        knorm = torch.linalg.vector_norm(keys, dim=-1)
        valid = (qnorm > 1e-6) & (knorm > 1e-6)
        score = (query[:, None, :] * keys).sum(-1) / (qnorm * knorm).clamp_min(1e-12)
        valid &= torch.isfinite(score)
        row_ids = torch.nonzero(valid.any(dim=-1), as_tuple=False).flatten()
        if not row_ids.numel():
            empty_labels = np.full(c, -1, np.int16)
            empty_starts = np.zeros((0, a.topm), np.int32)
            empty_weights = np.zeros((0, a.topm), np.float32)
            empty_counts = np.zeros((0,), np.uint8)
            return fallback.expand(h, c).clone(), c, (empty_labels, empty_starts, empty_weights, empty_counts)
        scores, row_valid = score[row_ids], valid[row_ids]
        pref = torch.softmax((scores / a.cluster_temperature).masked_fill(~row_valid, -torch.inf), dim=-1)
        k = min(a.groups, row_ids.numel())
        centers = torch.empty((k, pref.shape[1]), dtype=pref.dtype, device=self.device)
        centers[0] = pref[0]
        min_dist = ((pref - centers[0]) ** 2).sum(-1)
        for j in range(1, k):
            centers[j] = pref[torch.argmax(min_dist)]
            min_dist = torch.minimum(min_dist, ((pref - centers[j]) ** 2).sum(-1))
        for _ in range(a.kmeans_iters):
            labels = torch.cdist(pref.double(), centers.double(), p=2, compute_mode='donot_use_mm_for_euclid_dist').argmin(-1)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, labels, pref)
            counts = torch.bincount(labels, minlength=k)
            centers = torch.where((counts > 0)[:, None], sums / counts.clamp_min(1)[:, None], centers)
        labels = torch.cdist(pref.double(), centers.double(), p=2, compute_mode='donot_use_mm_for_euclid_dist').argmin(-1)
        counts = torch.bincount(labels, minlength=k)
        sums = torch.zeros((k, scores.shape[1]), device=self.device, dtype=scores.dtype)
        sums.index_add_(0, labels, scores)
        invalid = torch.zeros((k, scores.shape[1]), device=self.device, dtype=torch.int32)
        invalid.index_add_(0, labels, (~row_valid).to(torch.int32))
        common = (invalid == 0) & (counts[:, None] > 0)
        avg = (sums / counts.clamp_min(1)[:, None]).masked_fill(~common, -torch.inf)
        nsel = common.sum(-1).clamp(max=a.topm)
        m = min(a.topm, scores.shape[1])
        vals, chosen = torch.topk(avg, m, dim=-1, sorted=True)
                                                                                 
                                                                                    
        ties = (((avg[:, :, None] == vals[:, None, :]) &
                 torch.isfinite(vals[:, None, :])).sum(dim=1) > 1).any(dim=1)
        tie_ids = torch.nonzero(ties, as_tuple=False).flatten().cpu().numpy()
        if tie_ids.size:
            tie_avg = avg[ties].cpu().numpy()
            tie_nsel = nsel[ties].cpu().numpy()
            for index, group in enumerate(tie_ids):
                count = int(tie_nsel[index])
                if count:
                    selected = np.argpartition(tie_avg[index], -count)[-count:]
                    selected = selected[np.argsort(tie_avg[index][selected])[::-1]].copy()
                    chosen[int(group), :count] = torch.as_tensor(selected, device=self.device)
            vals = avg.gather(1, chosen)
        selection_mask = torch.arange(m, device=self.device)[None, :] < nsel[:, None]
        safe_vals = torch.where(selection_mask, vals / a.weight_temperature, torch.full_like(vals, -torch.inf))
        safe_vals = torch.where((nsel == 0)[:, None], torch.zeros_like(safe_vals), safe_vals)
        weights = torch.softmax(safe_vals, dim=-1) * selection_mask
        usable = nsel[labels] > 0
        result = fallback.expand(h, c).clone()
        times_all = starts[chosen] + l
        steps = torch.arange(h, device=self.device)
        for lo in range(0, row_ids.numel(), 256):
            hi = min(lo + 256, row_ids.numel())
            ids = row_ids[lo:hi]
            labs = labels[lo:hi]
            times = times_all[labs, :, None] + steps[None, None, :]
            values = series[times, ids[:, None, None]]
            if self.center_values:
                values = values - values[:, :, -1:]
            value = (weights[labs, :, None] * values).sum(dim=1).T
            result[:, ids] = torch.where(usable[lo:hi][None, :], value, result[:, ids])
        label_np = np.full(c, -1, np.int16)
        ids_np = row_ids.cpu().numpy()
        labels_np = labels.cpu().numpy()
        usable_np = usable.cpu().numpy()
        label_np[ids_np[usable_np]] = labels_np[usable_np]
        choices_np = chosen.cpu().numpy()
        weights_np = weights.cpu().numpy()
        nsel_np = nsel.cpu().numpy()
        starts_np = starts.cpu().numpy().astype(np.int32, copy=False)
        packed_starts = np.zeros((k, a.topm), dtype=np.int32)
        packed_weights = np.zeros((k, a.topm), dtype=np.float32)
        packed_counts = nsel_np.astype(np.uint8, copy=False)
        for j in range(k):
            n = int(nsel_np[j])
            if n:
                packed_starts[j, :n] = starts_np[choices_np[j, :n]]
                packed_weights[j, :n] = weights_np[j, :n]
                self.selected_counts[n] = self.selected_counts.get(n, 0) + 1
        return result, int(c - usable.sum().item()), (label_np, packed_starts, packed_weights, packed_counts)

    @torch.no_grad()
    def apply_metadata(self, series, offset, is_trend, metadata):
        labels, packed_starts, packed_weights, packed_counts = metadata
        c = series.shape[1]
        fallback = (torch.zeros((1, c), device=self.device, dtype=series.dtype)
                    if self.center_values else
                    (offset.clone() if is_trend else torch.zeros((1, c), device=self.device, dtype=series.dtype)))
        if not packed_counts.size or not np.any(labels >= 0):
            return fallback.expand(self.h, c).clone()
        labels_t = torch.as_tensor(labels.astype(np.int64, copy=False), device=self.device)
        starts_t = torch.as_tensor(packed_starts.astype(np.int64, copy=False), device=self.device)
        weights_t = torch.as_tensor(packed_weights, device=self.device, dtype=series.dtype)
        counts_t = torch.as_tensor(packed_counts.astype(np.int64, copy=False), device=self.device)
        valid = labels_t >= 0
        groups = labels_t.clamp(min=0, max=packed_counts.size - 1)
        valid &= counts_t[groups] > 0
        channels = torch.arange(c, device=self.device)
        starts = starts_t[groups]
        weights = weights_t[groups]
        times = starts[:, :, None] + self.l + torch.arange(self.h, device=self.device)[None, None, :]
        values = series[times, channels[:, None, None]]
        if self.center_values:
            values = values - values[:, :, -1:]
        forecast = (weights[:, :, None] * values).sum(dim=1).transpose(0, 1)
        return torch.where(valid[None, :], forecast, fallback.expand(self.h, c))

    @torch.no_grad()
    def retrieve(self, indices, x):
        out_t, out_s = [], []
        for n, q in enumerate(indices.tolist()):
            if q in self.cache:
                tmeta, smeta = self.cache[q]
                offset = x[n, -1:, :]
                out_t.append(self.apply_metadata(self.trend, offset, True, tmeta))
                out_s.append(self.apply_metadata(self.season, offset, False, smeta))
                continue
            starts = self.starts[self.starts + self.l + self.h <= q]
            if starts.numel() > self.args.max_candidates:
                ix = torch.linspace(0, starts.numel()-1, self.args.max_candidates, device=self.device).long()
                starts = starts[ix]
            self.queries += 1
            self.no_candidate += int(not starts.numel())
            offset = x[n, -1:, :]
            t, ft, tmeta = self.component(self.trend, q, starts, offset, True)
            s, fs, smeta = self.component(self.season, q, starts, offset, False)
            self.channel_fallback += ft + fs
            entry_bytes = sum(arr.nbytes for meta in (tmeta, smeta) for arr in meta)
            if self.cache_bytes + entry_bytes <= self.args.cache_mb * 1024**2:
                self.cache[q] = (tmeta, smeta)
                self.cache_bytes += entry_bytes
            out_t.append(t); out_s.append(s)
        return torch.stack(out_t), torch.stack(out_s), None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--pred-len', type=int, required=True)
    p.add_argument('--seq-factor', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=None, help='Explicit input length; overrides seq-factor')
    p.add_argument('--freq', default='h')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--learning-rate', type=float, default=0.001)
    p.add_argument('--groups', type=int, default=4)
    p.add_argument('--topm', type=int, default=20)
    p.add_argument('--max-candidates', type=int, default=32)
    p.add_argument('--candidate-stride', type=int, default=7)
    p.add_argument('--kmeans-iters', type=int, default=3)
    p.add_argument('--ema-alpha', type=float, default=0.2)
    p.add_argument('--cluster-temperature', type=float, default=0.1)
    p.add_argument('--weight-temperature', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--cache-mb', type=int, default=256)
    p.add_argument('--output', default='results/component_raft')
    p.add_argument('--model', choices=['ComponentRAFT', 'ComponentRAFTSeasonTrend', 'DLinear', 'iTransformer', 'DUET', 'PatchTST', 'UCast', 'TSMixer', 'CCM', 'ModernTCN'], default='ComponentRAFT')
    p.add_argument('--variant', choices=['original', 'retrieval'], default='retrieval')
    p.add_argument('--record-queries', type=int, default=0, help='Deprecated: use the separate visualization script')
    p.add_argument('--plot-channel', type=int, default=0)
    p.add_argument('--d-model', type=int, default=512)
    p.add_argument('--max-train-batches', type=int, default=0, help='Smoke validation only')
    p.add_argument('--max-eval-batches', type=int, default=0, help='Smoke validation only')
    p.add_argument('--checkpoint-in', default=None, help='Evaluate an existing model checkpoint without training')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a = p.parse_args()
    if a.record_queries:
        p.error('Visualization is separate: use scripts/visualize_component_retrieval.py')
    if not (0 < a.ema_alpha <= 1 and a.groups > 0 and a.topm > 0 and a.max_candidates > 0 and a.candidate_stride > 0 and a.cluster_temperature > 0 and a.weight_temperature > 0):
        p.error('Invalid retrieval parameters')
    if a.max_candidates < a.topm:
        p.error('max-candidates must be at least topm')
    if a.model in ('ComponentRAFT', 'ComponentRAFTSeasonTrend') and a.variant == 'original':
        p.error(f'{a.model} is the retrieval model, not an original backbone')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device)
    l, h = (a.seq_len if a.seq_len is not None else a.pred_len*a.seq_factor), a.pred_len
    cfg = SimpleNamespace(data=a.data, local_dataset_root=str(ROOT/'data/High_Dimensional_Time_Series'), root_path='', data_path='', features='M', target='OT', seq_len=l, label_len=0, pred_len=h, embed='timeF', freq=a.freq, seasonal_patterns='Monthly', batch_size=a.batch_size, num_workers=0, augmentation_ratio=0, model=a.model, task_name='long_term_forecast', moving_avg=25, d_model=a.d_model, d_ff=a.d_model, n_heads=8, e_layers=2, factor=1, dropout=0.1, activation='gelu')
    cfg.alpha = 0.001
    cfg.channel_reduction_ratio = 16
    cfg.n_cluster = 10
    cfg.accelerator = SimpleNamespace(device=device)
    model_config = ROOT / 'configs' / (a.model + '.yaml')
    if model_config.exists():
        with model_config.open(encoding='utf8') as f:
            defaults = (yaml.safe_load(f) or {}).get(a.data, {}) or {}
                                                                            
                                                                           
        for key in ('d_model', 'd_ff', 'n_heads', 'e_layers', 'n_cluster',
                    'channel_reduction_ratio', 'alpha', 'moving_avg', 'dropout'):
            if key in defaults:
                setattr(cfg, key, defaults[key])
    datasets = [data_provider(cfg, flag)[0] for flag in ('train', 'val', 'test')]
    train, val, test = datasets
    origin = [0, len(train.data_x)-l, len(train.data_x)+len(val.data_x)-2*l]
    data = np.concatenate((train.data_x, val.data_x[l:], test.data_x[l:]), axis=0).astype(np.float32)
    if not np.isfinite(data).all():
        raise ValueError('Dataset contains nonfinite values; explicit missing-data policy required')
    cfg.enc_in = data.shape[1]
    print(json.dumps({'data':a.data,'channels':cfg.enc_in,'seq_len':l,'pred_len':h,'split_sizes':[len(d) for d in datasets], 'device':str(device)}, ensure_ascii=False), flush=True)
    is_component_model = a.model in ('ComponentRAFT', 'ComponentRAFTSeasonTrend')
    retriever = Retrieval(data, len(train.data_x), l, h, a, device,
                          center_values=(a.model != 'ComponentRAFT')) if a.variant == 'retrieval' else None
    loaders = [DataLoader(Indexed(d, o), batch_size=a.batch_size, shuffle=(i==0), num_workers=0) for i,(d,o) in enumerate(zip(datasets, origin))]
    if a.model in ('DUET', 'CCM'):
        import importlib
        model = importlib.import_module('models.' + a.model).Model(cfg)
    else:
        model = model_manager.create_model(a.model, cfg)
    if not is_component_model and a.variant == 'retrieval':
        from models.ComponentForecastAdapter import ComponentForecastAdapter
        model = ComponentForecastAdapter(model, h, centered_references=True)
    model = model.to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
                                                                             
                                              
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    optim = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    stem = a.data.replace(' ', '_') + f'_L{l}_H{h}'
    checkpoint = out/(stem+'.pt')
    best = float('inf'); best_epoch = 0; stale = 0; history = []
    def run(loader, training, phase='val'):
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        model.train(training); sq = ab = count = 0
        samples = 0
        with torch.set_grad_enabled(training):
            for batch_no, (x, y, xmark, idx) in enumerate(loader, 1):
                x = x.to(device=device, dtype=torch.float32)
                y = y[:, -h:, :].to(device=device, dtype=torch.float32)
                marks = xmark.to(device=device, dtype=torch.float32)
                if retriever is None:
                    pred, auxiliary = model(x, marks, None, None)
                else:
                    refs = retriever.retrieve(idx, x)
                    pred, auxiliary = model(x, refs, marks) if not is_component_model else model(x, refs)
                err = pred-y
                loss = err.square().mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                if training:
                    optim.zero_grad(); (loss + auxiliary).backward(); optim.step()
                sq += err.detach().square().sum().item(); ab += err.detach().abs().sum().item(); count += err.numel()
                samples += x.shape[0]
                if batch_no % 100 == 0:
                    print(json.dumps({'phase': phase, 'batch': batch_no, 'total_batches': len(loader)}), flush=True)
                limit = a.max_train_batches if training else a.max_eval_batches
                if limit and batch_no >= limit: break
        if device.type == 'cuda': torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        peak_allocated = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None
        return {'mse':sq/count, 'mae':ab/count, 'seconds':seconds,
                'samples':samples, 'samples_per_second':samples/seconds,
                'peak_allocated_bytes':peak_allocated,
                'peak_reserved_bytes':peak_reserved}
    start = time.time()
    if a.checkpoint_in:
        checkpoint = Path(a.checkpoint_in)
        best, best_epoch = None, None
    else:
        for epoch in range(1, a.epochs+1):
            tr = run(loaders[0], True, 'train'); va = run(loaders[1], False, 'val')
            history.append({'epoch':epoch,'train':tr,'val':va})
            print(json.dumps(history[-1]), flush=True)
            if va['mse'] < best:
                best = va['mse']; best_epoch = epoch; stale = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
                if stale >= a.patience: break
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    cache_bytes_at_test_start = retriever.cache_bytes if retriever else 0
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    te = run(loaders[2], False, 'test')
    performance = {'test_seconds_including_retrieval':te['seconds'], 'test_samples_per_second':te['samples_per_second'],
                   'test_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                   'test_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device.type == 'cuda' else None,
                   'test_cache_bytes':retriever.cache_bytes if retriever else 0, 'device':str(device),
                   'cache_bytes_at_test_start':cache_bytes_at_test_start,
                   'timing_scope':'data loading, retrieval, forecast and metrics; excludes plotting and data preparation'}
    arch = {k: v for k, v in vars(cfg).items() if isinstance(v, (str, int, float, bool))}
    result = {'config':vars(a),'architecture_config':arch,'parameter_count':parameter_count,'channels':cfg.enc_in,'seq_len':l,'pred_len':h,'split_sizes':[len(d) for d in datasets], 'best_epoch':best_epoch,'val_mse':best,'test':te,'performance':performance,'history':history,'seconds':time.time()-start,'retrieval_queries':retriever.queries if retriever else 0,'no_candidate_queries':retriever.no_candidate if retriever else 0,'component_channel_fallbacks':retriever.channel_fallback if retriever else 0,'selected_count_histogram':retriever.selected_counts if retriever else {},'checkpoint':str(checkpoint)}
    (out/(stem+'.json')).write_text(json.dumps(result, indent=2), encoding='utf8')
    print('RESULT '+json.dumps(result, ensure_ascii=False), flush=True)

if __name__ == '__main__': main()
