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


from .npz_block_dataloader import NpzBlockDataLoader
from .npz_dataloader import NpzDataLoader
from .parquet_block_dataloader import ParquetBlockDataLoader
from .parquet_dataloader import ParquetDataLoader
import logging


def _log_dataset_stats(split_name, data_gen):
    num_samples = getattr(data_gen, "num_samples", None)
    num_blocks = getattr(data_gen, "num_blocks", None)
    if num_samples is None or num_samples < 0:
        logging.info("%s samples: total/unknown, blocks/%s", split_name, num_blocks)
    else:
        logging.info("%s samples: total/%d, blocks/%d", split_name, num_samples, num_blocks)


class RankDataLoader(object):
    def __init__(self, feature_map, stage="both", train_data=None, valid_data=None, test_data=None,
                 batch_size=32, shuffle=True, streaming=False, data_format="npz", **kwargs):
        logging.info("Loading datasets...")
        train_gen = None
        valid_gen = None
        test_gen = None
        data_format = (data_format or "npz").lower()
        if kwargs.get("data_loader"):
            DataLoader = kwargs["data_loader"]
        else:
            if data_format == "npz":
                DataLoader = NpzBlockDataLoader if streaming else NpzDataLoader
            elif data_format in ["tfrecord", "tf_record"]:
                from .tfrecord_dataloader import TFRecordDataLoader
                DataLoader = TFRecordDataLoader
            else: # ["parquet", "csv"]
                DataLoader = ParquetBlockDataLoader if streaming else ParquetDataLoader
        self.stage = stage
        if stage in ["both", "train"]:
            train_gen = DataLoader(feature_map, train_data, split="train", batch_size=batch_size,
                                   shuffle=shuffle, **kwargs)
            _log_dataset_stats("Train", train_gen)
            if valid_data:
                valid_gen = DataLoader(feature_map, valid_data, split="valid",
                                       batch_size=batch_size, shuffle=False, **kwargs)
                _log_dataset_stats("Validation", valid_gen)

        if stage in ["both", "test"]:
            if test_data:
                test_gen = DataLoader(feature_map, test_data, split="test", batch_size=batch_size,
                                      shuffle=False, **kwargs)
                _log_dataset_stats("Test", test_gen)
        self.train_gen, self.valid_gen, self.test_gen = train_gen, valid_gen, test_gen

    def make_iterator(self):
        if self.stage == "train":
            logging.info("Loading train and validation data done.")
            return self.train_gen, self.valid_gen
        elif self.stage == "test":
            logging.info("Loading test data done.")
            return self.test_gen
        else:
            logging.info("Loading data done.")
            return self.train_gen, self.valid_gen, self.test_gen
