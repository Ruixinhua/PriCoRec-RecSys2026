# PriCoRec

Research code for **“PriCoRec: A Privacy-Aware Cloud–Device Collaborative
Framework for Ad Recommendation under Feature Constraints.”**

PriCoRec was accepted as a Main Conference Short Paper at
[RecSys '26](https://recsys.acm.org/recsys26/), the 20th ACM Conference on
Recommender Systems, which will be held in Minneapolis, Minnesota, USA,
September 28–October 2, 2026.

The runnable path implements the following three-stage workflow:

1. cloud retrieval over cloud-accessible item and behavior features;
2. cloud PNN pre-ranking over `FG1+FG2`, with a DPP-inspired diversity
   regularizer; and
3. device-stage PNN re-ranking over `FG1+FG2+FG3`, using the frozen cloud logit
   as an auxiliary score.

The synthetic smoke test runs all three logical stages in one local Python
process. “Cloud” and “device” therefore describe the experimental feature-flow
contract, not physically isolated services. This repository is not a recovered
or frozen snapshot of the runs that produced the paper tables.

Production serving infrastructure, platform orchestration, production-specific
configuration, credentials, datasets, checkpoints, experiment logs, and
generated results are intentionally excluded.

## Release status

| Surface | Included | Verified in this release |
| --- | --- | --- |
| `FG1+FG2` PNN pre-ranking with diversity regularization | Yes | Synthetic CPU smoke run |
| `FG1+FG2+FG3` PNN re-ranking with cloud-logit input | Yes | Synthetic CPU smoke run |
| Synthetic data generation and preprocessing | Yes | Tests and synthetic CPU smoke run |
| Paper-aligned TaobaoAd and Ali-CCP configurations | Yes | YAML loading and configuration-contract tests |

The smoke configuration verifies that the maintained public code path runs. It
does **not** reproduce or validate the numerical tables in the paper.
See [Reproducibility scope](REPRODUCIBILITY.md) for the exact evidence boundary.

## Installation

The package requires Python 3.10 or newer; CI verifies Python 3.10 and 3.12. A
virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Use `python -m pip install -e .` for runtime dependencies only. The normal
Parquet/CSV workflow does not require TensorFlow. Install the optional TFRecord
reader only when needed:

```bash
python -m pip install -e ".[tfrecord]"
```

## Quick start

Run the complete synthetic workflow from the repository root on CPU:

```bash
python -m cloud_device_recsys.tools.generate_smoke_data

python -m cloud_device_recsys.run_preprocess \
  --raw_data_root ./data/raw/SmokeTest \
  --dataset_id SmokeTest \
  --config_dir ./cloud_device_recsys/config \
  --output_dir ./data/processed/SmokeTest \
  --force_rebuild

python -m cloud_device_recsys.run_pipeline \
  --config ./cloud_device_recsys/config \
  --pipeline_id pipeline_config/smoke_pipeline \
  --dataset_id SmokeTest \
  --mode full \
  --gpu -1 \
  --output_dir ./outputs/smoke \
  --experiment_id pricorec_smoke \
  --save_stage_outputs
```

The final metrics are written to
`outputs/smoke/pricorec_smoke/metrics.json`. Generated `data/` and `outputs/`
directories are ignored by Git.

Run the automated checks with:

```bash
python -m pytest -q
```

## Paper configurations

The current repository includes the paper-aligned dataset schemas and runnable
pipeline configurations for the two publicly downloadable datasets. Each
pipeline encodes the paper's `1,000 -> 100 -> 10` cascade, cloud-side
`FG1+FG2` boundary, device-side `FG1+FG2+FG3` boundary, 1:4 pre-ranking and
ranking negative sampling, 32-dimensional cloud PNN, 4-dimensional device PNN,
and the reported per-dataset diversity weight `lambda = 1e-2`.

| Dataset | Pipeline ID | Reported pre-ranking gAUC / R@100 | Reported device gAUC / R@1 / R@10 / N@10 / RR |
| --- | --- | --- | --- |
| Taobao Display Ad Click | [`pipeline_config/taobaoad_paper`](cloud_device_recsys/config/pipeline_config/taobaoad_paper.yaml) | 89.36 / 68.29 | 86.93 / 4.14 / 21.74 / 11.60 / 10.11 |
| Ali-CCP | [`pipeline_config/ali_ccp_paper`](cloud_device_recsys/config/pipeline_config/ali_ccp_paper.yaml) | 73.69 / 47.72 | 73.30 / 7.06 / 24.10 / 16.02 / 13.43 |

After obtaining a dataset, normalize it to the field names and three-split CSV
layout declared in
[`cloud_device_recsys/config/dataset_config.yaml`](cloud_device_recsys/config/dataset_config.yaml).
Then run, for example:

```bash
DATASET_ID=TaobaoAd
PIPELINE_ID=pipeline_config/taobaoad_paper
RAW_DATA_ROOT=/path/to/normalized/TaobaoAd

python -m cloud_device_recsys.run_preprocess \
  --raw_data_root "$RAW_DATA_ROOT" \
  --dataset_id "$DATASET_ID" \
  --config_dir ./cloud_device_recsys/config \
  --output_dir "./data/processed/$DATASET_ID" \
  --force_rebuild

python -m cloud_device_recsys.run_pipeline \
  --config ./cloud_device_recsys/config \
  --pipeline_id "$PIPELINE_ID" \
  --dataset_id "$DATASET_ID" \
  --mode full \
  --gpu 0 \
  --output_dir "./outputs/paper/$DATASET_ID" \
  --experiment_id "${DATASET_ID}_paper"
```

For Ali-CCP, use `DATASET_ID=Ali_CCP` and
`PIPELINE_ID=pipeline_config/ali_ccp_paper`. Set `--gpu -1` for CPU execution;
the full public datasets are intended for a suitably provisioned accelerator.

## Feature-flow contract

The schemas in
[`cloud_device_recsys/config/dataset_config.yaml`](cloud_device_recsys/config/dataset_config.yaml)
use:

- `FG1`: cloud-accessible candidate-item features;
- `FG2`: cloud-accessible behavior-sequence features; and
- `FG3`: private user/device features reserved for the logical device stage.

Every dataset field receives an explicit, reviewed `feature_group`; the paper
configs do not rely on a default assignment or field-name heuristic for the
cloud/device boundary. The smoke pipeline remains a synthetic integration
fixture, while `taobaoad_paper.yaml` and `ali_ccp_paper.yaml` encode the
paper-aligned public-dataset settings.

Feature separation is an experimental information-flow constraint. It does not,
by itself, establish differential privacy, secure aggregation, legal
compliance, or production isolation.

## Data

Datasets are not redistributed. Researchers must obtain each dataset from its
owner and comply with its license and terms:

- [Taobao Display Ad Click dataset (Tianchi)](https://tianchi.aliyun.com/dataset/56?lang=en-us)
- [Ali-CCP data page (Tianchi, dataId 408)](https://tianchi.aliyun.com/dataset/dataDetail?dataId=408)

An official public download source for the OpenMCC variant used in the study was
not verified for this release, so this repository does not provide an
unverified mirror or substitute link. The included TaobaoAd and Ali-CCP YAML
files expect normalized local splits and do not redistribute source data. See
[`DATA.md`](DATA.md) for the input contract and adaptation checklist.

## Repository layout

```text
cloud_device_recsys/   Data, models, three-stage pipeline, and runnable tools
fuxictr/               Modified FuxiCTR 2.3.9 subset required by the pipeline
model_zoo/             PNN and research comparison backbones/adapters
tests/                 Unit and integration tests for the public code path
DATA.md                Input schema and dataset adaptation checklist
REPRODUCIBILITY.md     Evidence boundary for numerical reproduction claims
```

The comparison adapters under `model_zoo/PrivacyPreserving/` are research
interfaces. `DPSGD` is fail-closed because this release does not include a
verified per-sample gradient implementation and privacy accountant. The
federated-style adapters do not provide federated transport or secure
aggregation.

## Citation

Please cite the RecSys '26 paper and this software release. Machine-readable
metadata is available in [`CITATION.cff`](CITATION.cff), including the venue and
current in-press status. The DOI, page range, and final proceedings metadata
will be added once ACM publishes them.

## License and attribution

PriCoRec is released under the [Apache License 2.0](LICENSE). The repository
contains a modified Apache-2.0-licensed subset of FuxiCTR. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution and the fixed
upstream revision.
