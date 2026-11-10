from .data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader
from pathlib import Path
import yaml

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom
}


def data_provider(args, flag, accelerator=None):
    if args.data in data_dict.keys():
        Data = data_dict[args.data]
    else:
        Data = data_dict['custom']

    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    root_path, data_path = args.root_path, args.data_path
    local_root = getattr(args, 'local_dataset_root', None)
    if local_root:
        root = Path(local_root)
        manifest = root / 'datasets.yaml'
        if not manifest.is_file():
            raise FileNotFoundError(f'Local dataset manifest is missing: {manifest}')
        entries = (yaml.safe_load(manifest.read_text(encoding='utf-8')) or {}).get('configs', [])
        names = {item['config_name']: item['data_files'] for item in entries}
        names['Electricity'] = names.get('ECL', 'electricity.csv')
        if args.data not in names:
            raise ValueError(f'Dataset {args.data!r} is not in {manifest}')
        root_path, data_path = str(root), names[args.data]
        if not (root / data_path).is_file():
            raise FileNotFoundError(f'Local dataset file missing: {root / data_path}')

    data_set = Data(
        args = args,
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )

    if accelerator:
        accelerator.print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True)
    
    return data_set, data_loader 
