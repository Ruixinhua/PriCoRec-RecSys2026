# Modified by the PriCoRec authors in 2026.
# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import os
import logging
import numpy as np
import gc
import multiprocessing as mp
import polars as pl
import glob

# 设置日志级别为 DEBUG
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def print_dataset_info(ddf, name="Dataset"):
    """
    打印数据集的基本统计信息
    Args:
        ddf: polars LazyFrame 或 DataFrame
        name: 数据集名称
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"数据集统计信息 - {name}")
    logging.info(f"{'='*60}")

    try:
        # 如果是LazyFrame，获取schema信息
        if isinstance(ddf, pl.LazyFrame):
            schema = ddf.collect_schema()
            logging.info(f"数据集行数: {ddf.select(pl.count()).collect().item()}")
            logging.info(f"数据集列数: {len(schema)}")

            # 输出列信息和数据类型
            logging.info(f"\n列信息:")
            for col_name, dtype in schema.items():
                logging.info(f"  {col_name}: {dtype}")

            # 数值列的基本统计信息
            numeric_cols = [col for col, dtype in schema.items()
                          if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                                     pl.Float32, pl.Float64]]

            if numeric_cols:
                logging.info(f"\n数值列统计信息:")
                for col in numeric_cols[:5]:  # 只显示前5个数值列，避免输出过长
                    try:
                        stats = ddf.select([
                            pl.col(col).min().alias("min"),
                            pl.col(col).max().alias("max"),
                            pl.col(col).mean().alias("mean"),
                            pl.col(col).std().alias("std")
                        ]).collect()

                        row = stats.row(0)
                        logging.info(f"  {col}: min={row[0]:.4f}, max={row[1]:.4f}, "
                                   f"mean={row[2]:.4f}, std={row[3]:.4f}")
                    except Exception as e:
                        logging.warning(f"  {col}: 无法计算统计信息 - {str(e)}")

                if len(numeric_cols) > 5:
                    logging.info(f"  ... 还有 {len(numeric_cols) - 5} 个数值列")

        else:  # DataFrame
            logging.info(f"数据集形状: {ddf.shape}")
            logging.info(f"数据集列数: {len(ddf.columns)}")

            # 输出列信息
            logging.info(f"\n列信息:")
            for col_name, dtype in zip(ddf.columns, ddf.dtypes):
                logging.info(f"  {col_name}: {dtype}")

            # 跳过缺失值统计以提高性能
            logging.info(f"\n缺失值统计: 已跳过以提高性能")

    except Exception as e:
        logging.error(f"计算数据集统计信息时出错: {str(e)}")

    logging.info(f"{'='*60}\n")


def merge_part_files(data_dir, filename, num_parts, saved_format="parquet"):
    """
    合并分批处理生成的part文件为单个文件
    Args:
        data_dir: 数据目录
        filename: 文件名前缀（如'train', 'valid', 'test'）
        num_parts: part文件数量
        saved_format: 保存格式（'parquet' 或 'tfrecord'）
    """
    logging.info(f"开始合并 {num_parts} 个part文件...")

    if saved_format == "parquet":
        # 查找所有part文件
        part_pattern = os.path.join(data_dir, f"{filename}/part_*.parquet")
        part_files = sorted(glob.glob(part_pattern))

        if not part_files:
            raise RuntimeError(
                f"Expected {num_parts} parquet part files for {filename}, found 0. "
                "Refusing to merge an incomplete dataset."
            )

        if len(part_files) != num_parts:
            raise RuntimeError(
                f"Expected {num_parts} parquet part files for {filename}, found {len(part_files)}. "
                "Refusing to merge an incomplete dataset."
            )

        logging.info(f"找到 {len(part_files)} 个part文件")

        # 使用polars合并文件
        try:
            # 读取所有part文件并合并
            dfs = []
            for part_file in part_files:
                logging.debug(f"读取part文件: {part_file}")
                df = pl.read_parquet(part_file)
                dfs.append(df)

            # 合并所有DataFrame
            merged_df = pl.concat(dfs)
            logging.info(f"合并后数据形状: {merged_df.shape}")

            # 保存合并后的文件
            merged_file = os.path.join(data_dir, f"{filename}.parquet")
            merged_df.write_parquet(merged_file)
            logging.info(f"合并文件保存到: {merged_file}")

            # 清理part文件和目录
            for part_file in part_files:
                os.remove(part_file)
                logging.debug(f"删除part文件: {part_file}")

            # 删除part文件目录
            part_dir = os.path.join(data_dir, filename)
            if os.path.exists(part_dir) and not os.listdir(part_dir):
                os.rmdir(part_dir)
                logging.info(f"删除空的part目录: {part_dir}")

        except Exception as e:
            logging.error(f"合并part文件时出错: {e}")
            raise

    elif saved_format == "tfrecord":
        # TFRecord文件合并逻辑
        import tensorflow as tf
        from fuxictr.tensorflow_utils import configure_tensorflow_cpu_only

        configure_tensorflow_cpu_only(tf)

        part_pattern = os.path.join(data_dir, f"{filename}/part_*.tfrecord")
        part_files = sorted(glob.glob(part_pattern))

        if not part_files:
            raise RuntimeError(
                f"Expected {num_parts} TFRecord part files for {filename}, found 0. "
                "Refusing to merge an incomplete dataset."
            )

        if len(part_files) != num_parts:
            raise RuntimeError(
                f"Expected {num_parts} TFRecord part files for {filename}, found {len(part_files)}. "
                "Refusing to merge an incomplete dataset."
            )

        merged_file = os.path.join(data_dir, f"{filename}.tfrecord")

        # 合并TFRecord文件
        with tf.io.TFRecordWriter(merged_file, options=tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
            for part_file in part_files:
                logging.debug(f"读取part文件: {part_file}")
                for record in tf.data.TFRecordDataset(part_file, compression_type="GZIP"):
                    writer.write(record.numpy())

        logging.info(f"TFRecord合并文件保存到: {merged_file}")

        # 清理part文件
        for part_file in part_files:
            os.remove(part_file)

        # 删除part文件目录
        part_dir = os.path.join(data_dir, filename)
        if os.path.exists(part_dir) and not os.listdir(part_dir):
            os.rmdir(part_dir)

    else:
        raise ValueError(f"不支持的文件格式: {saved_format}")

    logging.info(f"Part文件合并完成: {filename}.{saved_format}")


def split_train_test(train_ddf=None, valid_ddf=None, test_ddf=None, valid_size=0,
                     test_size=0, split_type="sequential"):
    num_samples = len(train_ddf)
    train_size = num_samples
    instance_IDs = np.arange(num_samples)
    if split_type == "random":
        np.random.shuffle(instance_IDs)
    if test_size > 0:
        if test_size < 1:
            test_size = int(num_samples * test_size)
        train_size = train_size - test_size
        test_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0:
        if valid_size < 1:
            valid_size = int(num_samples * valid_size)
        train_size = train_size - valid_size
        valid_ddf = train_ddf.loc[instance_IDs[train_size:], :].reset_index()
        instance_IDs = instance_IDs[0:train_size]
    if valid_size > 0 or test_size > 0:
        train_ddf = train_ddf.loc[instance_IDs, :].reset_index()
    return train_ddf, valid_ddf, test_ddf


def transform_block(feature_encoder, df_block, filename, saved_format="parquet"):
    logging.info(f"Starting transform_block for {filename}, block size: {len(df_block)}")
    df_block = feature_encoder.transform(df_block)
    data_path = os.path.join(feature_encoder.data_dir, f"{filename}.{saved_format}")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    logging.info(f"Saving data to {saved_format}: " + data_path)
    if saved_format == "parquet":
        logging.info(f"Writing parquet file to {data_path}")
        df_block.to_parquet(data_path, index=False, engine="pyarrow")
        logging.info(f"Successfully wrote parquet file to {data_path}")
    elif saved_format == "tfrecord":
        convert_to_tfrecord(feature_encoder, df_block, data_path)
    else:
        raise ValueError(f"Unsupported saved_format: {saved_format}")


def _normalise_sequence_for_tfrecord(value, max_len, padding="post", padding_value=0):
    """Return a fixed-width integer sequence for TFRecord serialization."""
    if max_len <= 0:
        raise ValueError(f"Sequence max_len must be positive, got {max_len}")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        values = []
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]

    if padding not in {"pre", "post"}:
        raise ValueError(f"Unsupported sequence padding mode: {padding}")

    values = [int(item) for item in values]
    if len(values) > max_len:
        values = values[-max_len:] if padding == "pre" else values[:max_len]

    padding_values = [int(padding_value)] * (max_len - len(values))
    return padding_values + values if padding == "pre" else values + padding_values


def convert_to_tfrecord(feature_encoder, df_block, data_path):
    import tensorflow as tf
    from fuxictr.tensorflow_utils import configure_tensorflow_cpu_only
    from tqdm import tqdm

    configure_tensorflow_cpu_only(tf)

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    feature_spec = {}
    sequence_specs = {}
    for feature in feature_encoder.feature_cols:
        if feature.get("active") is False:
            continue
        if feature["type"] == "numeric":
            feature_spec[feature["name"]] = tf.io.FixedLenFeature(1, tf.float32)
        elif feature["type"] in ["categorical", "meta"]:
            feature_spec[feature["name"]] = tf.io.FixedLenFeature(1, tf.int64)
        elif feature["type"] == "sequence":
            feature_spec[feature["name"]] = tf.io.FixedLenFeature([feature["max_len"]], tf.int64)
            sequence_specs[feature["name"]] = feature
        else:
            raise ValueError(f"Unsupported feature type: {feature['type']}")
    for label in feature_encoder.label_cols:
        feature_spec[label["name"]] = tf.io.FixedLenFeature(1, tf.float32)
    with tf.io.TFRecordWriter(data_path, options=options) as writer:
        for _, row in tqdm(df_block.iterrows(), total=len(df_block)):
            feature_dict = {}

            # 处理所有特征
            for feat, spec in feature_spec.items():
                if feat not in row.index:
                    raise KeyError(f"Required TFRecord field is missing from transformed data: {feat}")

                if feat in sequence_specs:
                    sequence_feature = sequence_specs[feat]
                    values = _normalise_sequence_for_tfrecord(
                        row[feat],
                        max_len=int(sequence_feature["max_len"]),
                        padding=sequence_feature.get("padding", "post"),
                        padding_value=sequence_feature.get(
                            "padding_value", sequence_feature.get("padding_idx", 0)
                        ),
                    )
                    feature_dict[feat] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=values)
                    )
                elif spec.dtype == tf.float32:
                    feature_dict[feat] = tf.train.Feature(
                        float_list=tf.train.FloatList(value=[float(row[feat])])
                    )
                elif spec.dtype == tf.int64:
                    feature_dict[feat] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[int(row[feat])])
                    )
                else:
                    raise ValueError(f"Unsupported dtype for feature {feat}")

            # 创建 TFRecord Example
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example.SerializeToString())


def transform(feature_encoder, ddf, filename, block_size=0, saved_format="parquet"):
    """
    内存友好的数据转换函数，避免一次性加载大量数据到内存
    """
    logging.info(f"Starting transform for {filename}")

    # 设置默认的分批大小来避免内存溢出
    if block_size == 0:
        # 根据可用内存自动调整批次大小
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        # 保守估计，每GB内存处理100k行数据
        auto_block_size = max(50000, int(available_memory_gb * 100000))
        logging.info(f"Auto-detected block_size: {auto_block_size} based on available memory")
        block_size = auto_block_size

    # 获取总行数用于进度跟踪
    try:
        total_rows = ddf.select(pl.len()).collect().item()
        logging.info(f"Total rows to process: {total_rows}")
    except Exception as e:
        logging.warning(f"Could not get total row count: {e}")
        total_rows = None

    # 分批处理避免内存溢出
    if block_size > 0 and total_rows and total_rows > block_size:
        logging.info(f"Using batch processing with block_size: {block_size}")

        # 并行处理多个批次. Keep every AsyncResult so a worker exception is
        # surfaced before we merge part files.
        pool = mp.Pool(max(1, mp.cpu_count() // 2))
        async_results = []
        block_id = 0
        try:
            # 分批次处理
            for offset in range(0, total_rows, block_size):
                current_block_size = min(block_size, total_rows - offset)
                logging.info(f"Processing batch {block_id}, offset: {offset}, size: {current_block_size}")

                try:
                    # 分批collect数据，避免内存溢出
                    df_batch = ddf.slice(offset, current_block_size).collect().to_pandas()
                    logging.info(f"Successfully loaded batch {block_id} to pandas, shape: {df_batch.shape}")

                    # 异步处理批次
                    async_results.append(pool.apply_async(
                        transform_block,
                        args=(feature_encoder, df_batch,
                              '{}/part_{:05d}'.format(filename, block_id),
                              saved_format)
                    ))

                    block_id += 1

                    # 手动释放内存
                    del df_batch
                    gc.collect()

                except Exception as e:
                    logging.error(f"Error processing batch {block_id}: {e}")
                    raise

            pool.close()
            for result in async_results:
                result.get()
            pool.join()
        except Exception:
            pool.terminate()
            pool.join()
            raise

        # 合并所有part文件为单个文件
        logging.info(f"Merging {block_id} part files into single {saved_format} file")
        merge_part_files(feature_encoder.data_dir, filename, block_id, saved_format)

    else:
        # 小数据集或未指定分批大小，直接处理
        logging.info("Processing entire dataset at once")
        try:
            pandas_df = ddf.collect().to_pandas()
            logging.info(f"Converted to pandas DataFrame, shape: {pandas_df.shape}")
            transform_block(feature_encoder, pandas_df, filename, saved_format=saved_format)
            del pandas_df
            gc.collect()
        except Exception as e:
            logging.error(f"Error processing entire dataset: {e}")
            # 如果内存不足，尝试自动分批处理
            if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                logging.info("Memory error detected, falling back to batch processing")
                return transform(feature_encoder, ddf, filename, block_size=50000, saved_format=saved_format)
            else:
                raise

    logging.info(f"Completed transform for {filename}")


def build_dataset(feature_encoder, train_data=None, valid_data=None, test_data=None,
                  valid_size=0, test_size=0, split_type="sequential", data_block_size=0,
                  rebuild_dataset=True, **kwargs):
    """ Build feature_map and transform data """
    if rebuild_dataset:
        feature_map_path = os.path.join(feature_encoder.data_dir, "feature_map.json")
        if os.path.exists(feature_map_path):
            logging.warn(f"Skip rebuilding {feature_map_path}. "
                + "Please delete it manually if rebuilding is required.")
        else:
            if train_data is None:
                raise ValueError(
                    f"feature_map.json not found at {feature_map_path} and 'train_data' is not "
                    f"set in the config. Either: (1) set 'train_data' path in dataset_config to "
                    f"build from raw data, or (2) ensure the preprocessed data (including "
                    f"feature_map.json) already exists at data_root/dataset_id/."
                )
            # Load data files
            train_ddf = feature_encoder.read_data(train_data, **kwargs)
            logging.info("Raw training data loaded successfully")
            print_dataset_info(train_ddf, "原始训练数据")

            valid_ddf = None
            test_ddf = None

            # Split data for train/validation/test
            if valid_size > 0 or test_size > 0:
                valid_ddf = feature_encoder.read_data(valid_data, **kwargs)
                if valid_ddf is not None:
                    print_dataset_info(valid_ddf, "原始验证数据")

                test_ddf = feature_encoder.read_data(test_data, **kwargs)
                if test_ddf is not None:
                    print_dataset_info(test_ddf, "原始测试数据")

                # TODO: check split_train_test in lazy mode
                train_ddf, valid_ddf, test_ddf = split_train_test(train_ddf, valid_ddf, test_ddf,
                                                                valid_size, test_size, split_type)

            # fit and transform train_ddf
            train_ddf = feature_encoder.preprocess(train_ddf)
            logging.info("Training data preprocessing completed")
            # print_dataset_info(train_ddf, "预处理后训练数据")

            feature_encoder.fit(train_ddf, rebuild_dataset=True, **kwargs)
            sf = kwargs.get("saved_format", "parquet")
            transform(feature_encoder, train_ddf, 'train', block_size=data_block_size, saved_format=sf)
            del train_ddf
            gc.collect()

            # Transfrom valid_ddf
            if valid_ddf is None and (valid_data is not None):
                valid_ddf = feature_encoder.read_data(valid_data, **kwargs)
                if valid_ddf is not None:
                    print_dataset_info(valid_ddf, "原始验证数据")

            if valid_ddf is not None:
                valid_ddf = feature_encoder.preprocess(valid_ddf)
                logging.info("Validation data preprocessing completed")
                print_dataset_info(valid_ddf, "预处理后验证数据")
                transform(feature_encoder, valid_ddf, 'valid', block_size=data_block_size, saved_format=sf)
                del valid_ddf
                gc.collect()

            # Transfrom test_ddf
            if test_ddf is None and (test_data is not None):
                test_ddf = feature_encoder.read_data(test_data, **kwargs)
                if test_ddf is not None:
                    print_dataset_info(test_ddf, "原始测试数据")

            if test_ddf is not None:
                test_ddf = feature_encoder.preprocess(test_ddf)
                logging.info("Test data preprocessing completed")
                print_dataset_info(test_ddf, "预处理后测试数据")
                transform(feature_encoder, test_ddf, 'test', block_size=data_block_size, saved_format=sf)
                del test_ddf
                gc.collect()
            logging.info("Transform csv data to parquet done.")

        train_data, valid_data, test_data = (
            os.path.join(feature_encoder.data_dir, "train"), \
            os.path.join(feature_encoder.data_dir, "valid"), \
            os.path.join(feature_encoder.data_dir, "test") if (
                test_data or test_size > 0) else None
        )

    else: # skip rebuilding data but only compute feature_map.json
        feature_encoder.fit(train_ddf=None, rebuild_dataset=False, **kwargs)

    # Return processed data splits
    return train_data, valid_data, test_data
