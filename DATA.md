# Data contract and adaptation checklist

The repository distributes synthetic data generation code plus paper-aligned
schemas and pipeline YAML for TaobaoAd and Ali-CCP. It does not distribute the
paper datasets, transformed copies, split manifests, or checkpoints.

## Expected split layout

The preprocessing entry point expects three split names under one raw-data
directory:

```text
<raw_data_root>/train.csv
<raw_data_root>/valid.csv
<raw_data_root>/test.csv
```

Alternative formats supported by FuxiCTR require a matching dataset
configuration and are not exercised by the default smoke workflow.

The `SmokeTest` fixture demonstrates the minimum configuration surface:

- `label_col`: binary target definition;
- `impression_id_col`: stable request/group identifier;
- `user_id_col`: stable user identifier retained for grouped accounting;
- `item_id_col`: candidate item identifier;
- `item_pool.file`: item feature pool name;
- `feature_cols`: names, types, dtypes, sequence settings, and feature groups;
  and
- `raw_data_root` / `processed_data_root`: local input and output locations.

See
[`cloud_device_recsys/config/dataset_config.yaml`](cloud_device_recsys/config/dataset_config.yaml)
for the synthetic fixture and the normalized TaobaoAd and Ali-CCP field maps.

## Included paper-aligned configurations

The two public-dataset pipelines are selected with
[`pipeline_config/taobaoad_paper`](cloud_device_recsys/config/pipeline_config/taobaoad_paper.yaml)
and
[`pipeline_config/ali_ccp_paper`](cloud_device_recsys/config/pipeline_config/ali_ccp_paper.yaml).
Their schemas explicitly encode the paper feature boundary: TaobaoAd has 11
cloud-accessible and 8 device-only model fields; Ali-CCP has 14
cloud-accessible and 9 device-only model fields. Fields excluded by the paper
mapping remain metadata or `drop` fields rather than model inputs.

These files expect already normalized `train.csv`, `valid.csv`, and `test.csv`
inputs. They do not download either dataset or reconstruct the paper's split
manifests from the vendor archives.

## Required review before using a real dataset

1. Verify the dataset license and access terms.
2. Freeze checksums for the raw files and record the split construction rule.
3. Map every input field explicitly to `FG1`, `FG2`, or `FG3`; do not infer a
   privacy boundary from field names.
4. Confirm that cloud stages use only `FG1+FG2` and the logical device stage is
   the only stage allowed to consume `FG3`.
5. Keep vocabulary fitting, train-negative pools, and model selection confined
   to the training/validation protocol. Do not use test labels for fitting.
6. Record the resolved YAML, seed, code SHA, dependency environment, and output
   checksums for every reported result.

Sequence fields must define `max_len`, `padding`, `splitter`, and any shared
embedding or feature encoder required by the chosen model. The synthetic fixture
uses comma-separated histories and `layers.MaskedAveragePooling()`.

## Dataset sources

- [Taobao Display Ad Click dataset (Tianchi)](https://tianchi.aliyun.com/dataset/56?lang=en-us)
- [Ali-CCP data page (Tianchi, dataId 408)](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408)

No verified official public source for the study's OpenMCC variant is provided
in this release.
