from pathlib import Path

import pytest

from cloud_device_recsys.config.config_parser import ConfigParser


CONFIG_DIR = Path(__file__).resolve().parents[1] / "cloud_device_recsys" / "config"


def _feature_names_by_group(dataset_config):
    grouped = {"FG1": [], "FG2": [], "FG3": [], "drop": []}
    for feature in dataset_config["feature_cols"]:
        names = feature["name"]
        if not isinstance(names, list):
            names = [names]
        grouped[feature["feature_group"]].extend(names)
    return grouped


@pytest.mark.parametrize(
    (
        "pipeline_name",
        "dataset_id",
        "cloud_count",
        "device_only_count",
        "user_id_group",
    ),
    [
        ("taobaoad_paper.yaml", "TaobaoAd", 11, 8, "drop"),
        ("ali_ccp_paper.yaml", "Ali_CCP", 14, 9, "FG3"),
    ],
)
def test_paper_config_contract(
    pipeline_name, dataset_id, cloud_count, device_only_count, user_id_group
):
    parser = ConfigParser(str(CONFIG_DIR))
    full_config = parser.get_full_config(
        pipeline_path=CONFIG_DIR / "pipeline_config" / pipeline_name,
        dataset_path=CONFIG_DIR / "dataset_config.yaml",
    )

    pipeline = full_config["pipeline"]
    dataset = full_config["dataset"]
    grouped = _feature_names_by_group(dataset)
    stages = pipeline["stages"]

    assert dataset["dataset_id"] == dataset_id
    assert len(grouped["FG1"] + grouped["FG2"]) == cloud_count
    assert len(grouped["FG3"]) == device_only_count
    assert dataset["user_id_col"] in grouped[user_id_group]

    assert stages["retrieval"]["features"] == ["FG1", "FG2"]
    assert stages["retrieval"]["top_k"] == 1000

    preranking = stages["preranking"]
    assert preranking["model"] == "PNN"
    assert preranking["features"] == ["FG1", "FG2"]
    assert preranking["top_k"] == 100
    assert preranking["model_params"]["num_negatives"] == 4
    assert preranking["model_params"]["embedding_dim"] == 32
    assert len(preranking["model_params"]["hidden_units"]) == 3
    assert preranking["model_params"]["use_diversity_loss"] is True
    assert preranking["model_params"]["diversity_lambda"] == pytest.approx(0.01)
    assert preranking["model_params"]["diversity_theta"] == pytest.approx(0.0)

    reranking = stages["reranking"]
    assert reranking["model"] == "PNN"
    assert reranking["features"] == ["FG1", "FG2", "FG3"]
    assert reranking["top_k"] == 10
    assert reranking["model_params"]["num_negatives"] == 4
    assert reranking["model_params"]["embedding_dim"] == 4
    assert len(reranking["model_params"]["hidden_units"]) == 2
    assert reranking["model_params"]["use_cloud_score"] is True
