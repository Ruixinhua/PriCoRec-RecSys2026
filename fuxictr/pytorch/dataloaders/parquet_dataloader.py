# Modified by the PriCoRec authors in 2026.
# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
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
import logging

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd

from fuxictr.utils import monitor_memory_in_realtime


class ParquetDataset(Dataset):
    def __init__(self, feature_map, data_path, low_memory=False):
        self.feature_map = feature_map
        self.darray = self.load_data(data_path, low_memory)

    def __getitem__(self, index):
        return self.darray[index, :]

    def __len__(self):
        return self.darray.shape[0]

    def load_data(self, data_path, low_memory=False):
        if low_memory:
            return self._load_data_low_memory(data_path)
        else:
            return self._load_data_standard(data_path)

    def _load_data_standard(self, data_path):
        """标准模式：一次性读取整个文件到内存"""
        df = pd.read_parquet(data_path)
        # print("Finished reading data from:", data_path)
        # monitor_memory_in_realtime()
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        missing_cols = [col for col in all_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(
                f"Parquet data is missing required feature/label columns: {missing_cols}. "
                "In particular, impression_id must be persisted by preprocessing; "
                "row indices are not valid request identifiers."
            )
        data_arrays = []
        for col in all_cols:
            if df[col].dtype == "object":
                array = np.array(df[col].to_list())
            else:
                array = df[col].to_numpy()
            data_arrays.append(array)
        import gc
        del df
        gc.collect()
        # monitor_memory_in_realtime()
        # print("Removed dataframe from memory")
        return np.column_stack(data_arrays)

    def _load_data_low_memory(self, data_path):
        """低内存模式：逐行组读取数据，复用标准模式的数据处理逻辑"""
        import pyarrow.parquet as pq
        logging.info(f"Reading data from: {data_path} (low memory mode)")

        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        parquet_file = pq.ParquetFile(data_path)
        available_cols = set(parquet_file.schema_arrow.names)
        missing_cols = [col for col in all_cols if col not in available_cols]
        if missing_cols:
            raise KeyError(
                f"Parquet data is missing required feature/label columns: {missing_cols}. "
                "In particular, impression_id must be persisted by preprocessing; "
                "row indices are not valid request identifiers."
            )

        # 获取总行数
        total_rows = parquet_file.metadata.num_rows

        # 第一遍：处理第一个行组来确定数组形状
        first_chunk = parquet_file.read_row_group(0, columns=all_cols)
        df_first = first_chunk.to_pandas()

        data_arrays = []
        for col in all_cols:
            if df_first[col].dtype == "object":
                array = np.array(df_first[col].to_list())
            else:
                array = df_first[col].to_numpy()
            data_arrays.append(array)

        first_chunk_array = np.column_stack(data_arrays)
        num_cols = first_chunk_array.shape[1]

        # 预分配最终数组
        final_array = np.empty((total_rows, num_cols), dtype=first_chunk_array.dtype)
        current_row = 0
        chunk_size = first_chunk_array.shape[0]

        # 复制第一个块的数据
        final_array[current_row:current_row + chunk_size] = first_chunk_array
        current_row += chunk_size

        # 清理第一个块的内存
        del first_chunk, df_first, data_arrays, first_chunk_array
        import gc
        gc.collect()

        # 处理剩余的行组
        for i in range(1, parquet_file.num_row_groups):
            # 读取当前行组并转换为 pandas DataFrame
            table_chunk = parquet_file.read_row_group(i, columns=all_cols)
            df_chunk = table_chunk.to_pandas()

            # 复用标准模式的数据处理逻辑
            data_arrays = []
            for col in all_cols:
                if df_chunk[col].dtype == "object":
                    array = np.array(df_chunk[col].to_list())
                else:
                    array = df_chunk[col].to_numpy()
                data_arrays.append(array)

            # 使用与标准模式相同的列合并逻辑
            chunk_array = np.column_stack(data_arrays)
            chunk_size = chunk_array.shape[0]

            # 直接写入预分配的数组
            final_array[current_row:current_row + chunk_size] = chunk_array
            current_row += chunk_size

            # 清理当前行组的内存
            del table_chunk, df_chunk, data_arrays, chunk_array
            import gc
            gc.collect()

        logging.info("Finished reading data and creating final array.")
        monitor_memory_in_realtime()
        return final_array


class ParquetDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False,
                 num_workers=1, low_memory=False, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        self.dataset = ParquetDataset(feature_map, data_path, low_memory)
        super().__init__(dataset=self.dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=BatchCollator(feature_map))
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches


class BatchCollator(object):
    def __init__(self, feature_map):
        self.feature_map = feature_map
        self.all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels

    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        batch_dict = dict()
        for col in self.all_cols:
            batch_dict[col] = batch_tensor[:, self.feature_map.get_column_index(col)]
        return batch_dict
