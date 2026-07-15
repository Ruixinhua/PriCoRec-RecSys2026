# Reproducibility scope

This release provides a maintained, executable implementation of the
paper-described stages. It does not provide a complete evidence package for
reproducing the numerical tables in the paper.

## What is verified

The public synthetic smoke workflow verifies, on CPU, that:

1. raw CSV data can be generated and preprocessed into the expected feature
   schema;
2. retrieval consumes the cloud-accessible feature groups;
3. PNN pre-ranking consumes `FG1+FG2` and executes the diversity-loss path; and
4. device-stage PNN re-ranking consumes `FG1+FG2+FG3` together with a frozen
   cloud logit.

The workflow is seeded, but exact metrics can still depend on dependency,
hardware, and numerical-kernel versions. Its synthetic metrics are integration
test outputs, not research results.

## What is not claimed

This release does not claim exact reproduction of any paper table. It does not
include the dataset snapshots, split manifests, frozen paper configurations,
trained checkpoints, or immutable run receipts needed to validate those
values. The included smoke YAML is a test fixture, not a recovered paper run
configuration.

For an exact result to be called reproduced, retain and publish one immutable
receipt containing at least:

- the Git commit SHA;
- dataset source, license, and content checksum;
- preprocessing version and split manifest;
- the complete resolved configuration and random seed;
- a dependency lock, `pip freeze`, or container digest;
- checkpoint checksum;
- raw predictions or sufficient evaluation inputs; and
- metric output with software and hardware metadata.

Results without that tuple should be reported as exploratory or unverified, not
as a reproduction of the paper.

## Run the synthetic verification

From a clean checkout:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q

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
  --experiment_id paper_smoke \
  --save_stage_outputs
```

Inspect `outputs/smoke/paper_smoke/metrics.json` and the stage outputs under the
same experiment directory. These generated files are intentionally ignored by
Git.

## Privacy and deployment boundary

All smoke stages run in one local process. This code does not ship a trusted
execution environment, encrypted transport, secure aggregation protocol,
privacy accountant, production access control, or compliance assessment. The
feature-group contract and comparison adapters must not be interpreted as
production privacy guarantees.
