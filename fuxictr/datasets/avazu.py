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


from fuxictr.preprocess import FeatureProcessor
from datetime import datetime, date
import polars as pl
import numpy as np


class CustomizedFeatureProcessor(FeatureProcessor):
    def convert_weekday(self, col_name=None):
        def _convert_weekday(timestamp):
            dt = date(int('20' + timestamp[0:2]), int(timestamp[2:4]), int(timestamp[4:6]))
            return int(dt.strftime('%w'))
        return pl.col("hour").map_elements(_convert_weekday, return_dtype=pl.Int32)

    def convert_weekend(self, col_name=None):
        def _convert_weekend(timestamp):
            dt = date(int('20' + timestamp[0:2]), int(timestamp[2:4]), int(timestamp[4:6]))
            return 1 if dt.strftime('%w') in ['6', '0'] else 0
        return pl.col("hour").map_elements(_convert_weekend, return_dtype=pl.Int32)

    def convert_hour(self, col_name=None):
        return pl.col("hour").map_elements(lambda x: int(x[6:8]), return_dtype=pl.Int32)

    def convert_to_bucket(self, col_name):
        def _convert_to_bucket(value):
            if value > 2:
                value = int(np.floor(np.log(value) ** 2))
            else:
                value = int(value)
            return value
        return pl.col(col_name).map_elements(_convert_to_bucket, return_dtype=pl.Int32).cast(pl.Int32)
