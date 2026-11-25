                                                                                
import argparse, json, subprocess, sys
from pathlib import Path
from ucast_paper_lengths import PAPER_LENGTHS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['DLinear','iTransformer','DUET','PatchTST','UCast','TSMixer','CCM']
STANDARD = {'ETTh1':'h','ETTh2':'h','ETTm1':'t','ETTm2':'t','Illness':'w','Weather':'t','ECL':'h','Traffic':'h','Exchange':'d','Solar':'t'}
HIGH = {'Atec':'t','Air Quality':'h','Temp':'h','Wind':'h','Traffic-CA':'h','Traffic-GBA':'h','Traffic-GLA':'h','Neurolib':'h','SIRS':'d','Meter':'t','SP500':'d','Wiki-20k':'d'}
HORIZONS = (96, 192, 336, 720)
ILLNESS_HORIZONS = (24, 36, 48, 60)

def topm_for(data, horizon):
    return 10

def build_rows(scope, datasets, models, seed):
    rows=[]
    for data in datasets:
        if data in HIGH:
            horizon, factor = PAPER_LENGTHS[data]
            windows = [(horizon*factor,horizon)]
            freq=HIGH[data]
        elif data == 'Illness':
            windows = [(512,h) for h in ILLNESS_HORIZONS]
            freq=STANDARD[data]
        elif data in STANDARD:
            windows = [(512,h) for h in ([336] if scope=='module' else HORIZONS)]
            freq=STANDARD[data]
        else:
            raise ValueError('No verified U-Cast paper window or standard protocol: '+data)
        for l,h in windows:
            for model in models:
                for variant in (['original','retrieval'] if scope=='module' else ['retrieval']):
                    rows.append(dict(scope=scope,data=data,model=model,variant=variant,seq_len=l,pred_len=h,freq=freq,topm=topm_for(data,h),max_candidates=32,candidate_stride=max(1,h//4),seed=seed))
    return rows

def audit(rows):
    groups={}
    for row in rows:
        assert row['topm']==topm_for(row['data'],row['pred_len']) and row['max_candidates']>=row['topm']
        if row['scope']=='module':
            key=(row['data'],row['model'],row['seq_len'],row['pred_len'],row['seed'])
            groups.setdefault(key,[]).append(row)
        if row['data'] in HIGH:
            h,f=PAPER_LENGTHS[row['data']]; assert (row['seq_len'],row['pred_len'])==(h*f,h)
        elif row['data'] == 'Illness':
            assert row['seq_len']==512 and row['pred_len'] in ILLNESS_HORIZONS
        else:
            assert row['seq_len']==512 and (row['pred_len']==336 if row['scope']=='module' else row['pred_len'] in (96,192,336,720))
    for pair in groups.values():
        assert len(pair)==2 and {x['variant'] for x in pair}=={'original','retrieval'}
        assert {k:v for k,v in pair[0].items() if k!='variant'}=={k:v for k,v in pair[1].items() if k!='variant'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scope',choices=['module','model'],required=True)
    p.add_argument('--datasets',nargs='+',default=list(STANDARD)+list(HIGH))
    p.add_argument('--models',nargs='+',choices=MODELS)
    p.add_argument('--seeds',type=int,nargs='+',default=[2021,2022,2023,2024,2025])
    p.add_argument('--output',default='results/paper_protocol')
    p.add_argument('--data-root',default=str(ROOT/'data/High_Dimensional_Time_Series'))
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true',help='skip completed results only when saved settings match')
    p.add_argument('--smoke',action='store_true',help='one batch/epoch; never publish as complete result')
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args()
    models=(a.models or MODELS) if a.scope=='module' else ['ComponentRAFTSeasonTrend']
    rows=[row for seed in a.seeds for row in build_rows(a.scope,a.datasets,models,seed)]; audit(rows)
    out=Path(a.output).resolve() / ('smoke' if a.smoke else 'full') / a.scope
    out.mkdir(parents=True,exist_ok=True)
    (out/'matrix.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    print(f'AUDIT OK: {len(rows)} runs; matrix: {out / "matrix.json"}',flush=True)
    if not a.execute: return
    failures=[]
    for row in rows:
        target=out/row['model']/row['variant']/row['data'].replace(' ','_')/f"L{row['seq_len']}_H{row['pred_len']}"/f"seed{row['seed']}"
        target.mkdir(parents=True,exist_ok=True)
        result_file=target/(row['data'].replace(' ','_')+f"_L{row['seq_len']}_H{row['pred_len']}.json")
        if a.resume and result_file.exists():
            previous=json.loads(result_file.read_text(encoding='utf8'))
            old=previous.get('config',{})
            expected={k:row[k] for k in ('data','model','variant','seq_len','pred_len','freq','topm','max_candidates','candidate_stride','seed')}
            expected.update(data_root=a.data_root, device=a.device, epochs=1 if a.smoke else (100 if row['data'] in HIGH else 10),
                            max_train_batches=1 if a.smoke else 0, max_eval_batches=1 if a.smoke else 0,
                            learning_rate=0.001, ema_alpha=0.2, cluster_temperature=0.1,
                            weight_temperature=0.1, kmeans_iters=3, patience=5 if row['data'] in HIGH else 3, cache_mb=256,
                            checkpoint_in=None)
            differs={k:(old.get(k),v) for k,v in expected.items() if old.get(k)!=v}
            batch=old.get('batch_size')
            valid_batches=[]
            b=a.batch_size
            while b not in valid_batches:
                valid_batches.append(b)
                if b == 1: break
                b=max(1,b//2)
            if batch not in valid_batches:
                differs['batch_size']=(batch,valid_batches)
            channels=previous.get('channels')
            expected_groups=(channels + (2 if channels < 100 else 9)) // (3 if channels < 100 else 10) if isinstance(channels,int) and channels>0 else None
            if old.get('groups') != expected_groups:
                differs['groups']=(old.get('groups'),expected_groups)
            checkpoint=previous.get('checkpoint')
            if differs or not checkpoint or not Path(checkpoint).is_file() or 'mse' not in previous.get('test',{}):
                raise ValueError(f'Existing result is incomplete or differs: {result_file}; differences={differs}')
            print('SKIP '+json.dumps(row,ensure_ascii=False),flush=True)
            continue
        cmd=[sys.executable,'-u',str(ROOT/'scripts/run_component_raft.py'),'--data',row['data'],'--data-root',a.data_root,'--model',row['model'],'--variant',row['variant'],'--seq-len',str(row['seq_len']),'--pred-len',str(row['pred_len']),'--freq',row['freq'],'--topm',str(row['topm']),'--max-candidates',str(row['max_candidates']),'--candidate-stride',str(row['candidate_stride']),'--seed',str(row['seed']),'--batch-size',str(a.batch_size),'--device',a.device,'--output',str(target)]
        cmd += ['--epochs',str(100 if row['data'] in HIGH else 10),'--patience',str(5 if row['data'] in HIGH else 3)]
        if a.smoke: cmd += ['--epochs','1','--max-train-batches','1','--max-eval-batches','1']
        print('RUN '+json.dumps(row,ensure_ascii=False),flush=True)
        batch = a.batch_size
        while True:
            cmd[cmd.index('--batch-size') + 1] = str(batch)
            with (target/'run.log').open('a',encoding='utf8') as log:
                log.write(f'\n=== batch size {batch} ===\n')
                log.flush()
                code=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
            log_text = (target/'run.log').read_text(encoding='utf8', errors='replace')
            attempt = log_text.rsplit(f'=== batch size {batch} ===', 1)[-1]
            if code == 0 or batch <= 1 or not ('out of memory' in attempt.lower() and 'cuda' in attempt.lower()):
                break
            batch = max(1, batch // 2)
            print(f'CUDA OOM; retrying batch size {batch}', flush=True)
        if code: failures.append(dict(row=row,returncode=code,log=str(target/'run.log')))
    (out/'failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf8')
    table_cmd=[sys.executable,str(ROOT/'scripts/summarize_standard_results.py'),
               '--input',str(out),'--output',str(out/'tables')]
    subprocess.run(table_cmd,cwd=ROOT,check=True)
    print(f'FINISHED: {len(rows)-len(failures)}/{len(rows)}; failures={len(failures)}',flush=True)
    if failures: sys.exit(1)

if __name__=='__main__': main()
