import gc
import random
import time
import json

import numpy as np
import polars as pl
from tqdm import tqdm

random.seed(2022)
np.random.seed(2022)
sample_skeleton_train_path = './sample_skeleton_train.csv'
common_features_train_path = './common_features_train.csv'
sample_skeleton_test_path = './sample_skeleton_test.csv'
common_features_test_path = './common_features_test.csv'
save_path = "./"
write_features_path = save_path + 'all_features'
sparse_columns = ['101', '121', '122', '124', '125', '126', '127', '128', '129', '205', '206', '207', '216', '508', '509', '702', '853', '301']
dense_columns = ['508', '509', '702', '853']
sparse_columns += ['109_14', '110_14', '127_14', '150_14', '210']
dense_columns += ['109_14', '110_14', '127_14', '150_14']
uses_columns = [col for col in sparse_columns] + \
    ['D' + col for col in dense_columns]
remap_id = True
max_seq = 30
seq_to_fea = {
    '109_14': '206', '110_14': '207', '127_14': '216', '150_14': '210'
}
vocabulary = dict(zip(sparse_columns, [{} for _ in range(len(sparse_columns))]))


def _append_feat(feat_dict, key, value):
    """Collect multiple feature values under the same field as a list."""
    existing = feat_dict.get(key)
    if existing is None:
        feat_dict[key] = value
    elif isinstance(existing, list):
        if max_seq and len(existing) <= max_seq:
            existing.append(value)
    else:
        feat_dict[key] = [existing, value]


def preprocess_data(mode='train'):
    assert mode in ['train', 'test']
    common_features_path = common_features_train_path if mode == "train" else common_features_test_path
    sample_skeleton_path = sample_skeleton_train_path if mode == "train" else sample_skeleton_test_path

    print(f"Start processing common_features_{mode}")
    common_feat_dict = {}
    with open(common_features_path, 'r') as fr:
        for line in tqdm(fr):
            line_list = line.strip().split(',')
            feat_strs = line_list[2]
            feat_dict = {}
            for fstr in feat_strs.split('\x01'):
                field, feat_val = fstr.split('\x02')
                feat, val = feat_val.split('\x03')
                if field in sparse_columns:
                    if remap_id:
                        alias = seq_to_fea.get(field, field)
                        mapped_id = vocabulary[alias].setdefault(feat, len(vocabulary[alias]))
                        _append_feat(feat_dict, field, str(mapped_id))
                    else:
                        _append_feat(feat_dict, field, feat)
                if field in dense_columns:
                    _append_feat(feat_dict, 'D' + field, val)
            common_feat_dict[line_list[0]] = feat_dict

    print('join feats...')
    with open(f"{write_features_path}_{mode}.tmp", 'w') as fw:
        fw.write('click,purchase,' + ','.join(uses_columns) + '\n')
        with open(sample_skeleton_path, 'r') as fr:
            for line in tqdm(fr):
                line_list = line.strip().split(',')
                if line_list[1] == '0' and line_list[2] == '1':
                    continue
                feat_strs = line_list[5]
                feat_dict = {}
                for fstr in feat_strs.split('\x01'):
                    field, feat_val = fstr.split('\x02')
                    feat, val = feat_val.split('\x03')
                    if field in sparse_columns:
                        if remap_id:
                            alias = seq_to_fea.get(field, field)
                            mapped_id = vocabulary[alias].setdefault(feat, len(vocabulary[alias]))
                            _append_feat(feat_dict, field, str(mapped_id))
                        else:
                            _append_feat(feat_dict, field, feat)
                    if field in dense_columns:
                        _append_feat(feat_dict, 'D' + field, val)
                feat_dict.update(common_feat_dict[line_list[3]])
                feats = line_list[1:3]
                for k in uses_columns:
                    value = feat_dict.get(k, '0')
                    if isinstance(value, list):
                        value = '^'.join(value)
                    feats.append(value)
                fw.write(','.join(feats) + '\n')

    print('encode feats...')
    with open(f"{write_features_path}.{mode}", 'w') as fw:
        fw.write('click,purchase,' + ','.join(uses_columns) + '\n')
        with open(f"{write_features_path}_{mode}.tmp", 'r') as fr:
            fr.readline()  # remove header
            for line in tqdm(fr):
                line_list = line.strip().split(',')
                new_line = line_list[:2]
                for value, feat in zip(line_list[2:], uses_columns):
                    new_line.append(value)
                fw.write(','.join(new_line) + '\n')


def reduce_mem(df: pl.DataFrame) -> pl.DataFrame:
    starttime = time.time()
    start_mem = df.estimated_size() / 1024**2
    cast_exprs = []
    for col_name, col_type in df.schema.items():
        if col_type not in {
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.Float16,
            pl.Float32,
            pl.Float64,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        }:
            continue
        col = df.get_column(col_name)
        c_min = col.min()
        c_max = col.max()
        if c_min is None or c_max is None:
            continue
        if col_type.is_integer():
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                cast_exprs.append(pl.col(col_name).cast(pl.Int8))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                cast_exprs.append(pl.col(col_name).cast(pl.Int16))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                cast_exprs.append(pl.col(col_name).cast(pl.Int32))
            else:
                cast_exprs.append(pl.col(col_name).cast(pl.Int64))
        elif col_type.is_float():
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                cast_exprs.append(pl.col(col_name).cast(pl.Float16))
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                cast_exprs.append(pl.col(col_name).cast(pl.Float32))
            else:
                cast_exprs.append(pl.col(col_name).cast(pl.Float64))

    if cast_exprs:
        df = df.with_columns(cast_exprs)
    end_mem = df.estimated_size() / 1024**2
    reduction = 0.0 if start_mem == 0 else 100 * (start_mem - end_mem) / start_mem
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem, reduction, (time.time() - starttime) / 60))
    gc.collect()
    return df


def split_frame(df: pl.DataFrame, test_size: float = 0.5, seed: int = 2022):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = np.arange(df.height)
    rng.shuffle(indices)
    split_idx = int(df.height * (1 - test_size))
    left_idx = indices[:split_idx]
    right_idx = indices[split_idx:]
    return df[left_idx], df[right_idx]


if __name__ == "__main__":
    preprocess_data(mode='train')
    preprocess_data(mode='test')
    train_data = reduce_mem(pl.scan_csv(f"{write_features_path}.train").collect())
    test_data = reduce_mem(pl.scan_csv(f"{write_features_path}.test").collect())
    val_data, test_data = split_frame(test_data, test_size=0.5, seed=2022)
    len_train_data = train_data.height
    len_val_data = val_data.height
    len_test_data = test_data.height
    print(f"train_data : {len_train_data}, val_data: {len_val_data}, test_data:{len_test_data}")
    print("start save all ")

    if remap_id:
        with open('vocabulary.json', 'w') as f:
            json.dump(vocabulary, f)
    train_data.write_csv(save_path + "train.csv")
    val_data.write_csv(save_path + "valid.csv")
    test_data.write_csv(save_path + "test.csv")
    print("complete")
