#!/usr/bin/env python
"""Generate a tiny raw CSV dataset for preprocessing and pipeline smoke tests."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


FIELDNAMES = [
    "impression_id",
    "label",
    "cand_item_id",
    "cand_category",
    "cand_brand",
    "cand_price_level",
    "clicked_item_seq",
    "clicked_category_seq",
    "user_id",
    "user_age_level",
    "user_gender",
    "device_type",
]


def _item_category(item_id: int) -> int:
    return (item_id % 6) + 1


def _item_brand(item_id: int) -> int:
    return (item_id % 5) + 1


def _item_price_level(item_id: int) -> int:
    return (item_id % 4) + 1


def _history(user_id: int, num_items: int, length: int = 6) -> list[int]:
    return [((user_id * 3 + offset) % num_items) + 1 for offset in range(length)]


def _make_row(
    split: str,
    row_idx: int,
    impression_offset: int,
    num_users: int,
    num_items: int,
    positive_only: bool,
    rng: random.Random,
) -> dict[str, object]:
    user_id = (row_idx % num_users) + 1
    history_items = _history(user_id, num_items)

    if positive_only or row_idx % 4 == 0:
        item_id = history_items[row_idx % len(history_items)]
        label = 1.0
    else:
        item_id = ((row_idx * 7 + user_id + rng.randint(0, num_items - 1)) % num_items) + 1
        label = 0.0

    history_categories = [_item_category(item_id) for item_id in history_items]

    return {
        "impression_id": impression_offset + row_idx,
        "label": label,
        "cand_item_id": item_id,
        "cand_category": _item_category(item_id),
        "cand_brand": _item_brand(item_id),
        "cand_price_level": _item_price_level(item_id),
        "clicked_item_seq": ",".join(str(x) for x in history_items),
        "clicked_category_seq": ",".join(str(x) for x in history_categories),
        "user_id": user_id,
        "user_age_level": (user_id % 6) + 1,
        "user_gender": (user_id % 2) + 1,
        "device_type": (user_id % 3) + 1,
    }


def _write_split(
    output_dir: Path,
    split: str,
    rows: int,
    impression_offset: int,
    num_users: int,
    num_items: int,
    positive_only: bool,
    rng: random.Random,
) -> Path:
    output_path = output_dir / f"{split}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row_idx in range(rows):
            writer.writerow(
                _make_row(
                    split=split,
                    row_idx=row_idx,
                    impression_offset=impression_offset,
                    num_users=num_users,
                    num_items=num_items,
                    positive_only=positive_only,
                    rng=rng,
                )
            )

    return output_path


def generate_smoke_data(
    output_dir: Path,
    train_rows: int,
    valid_rows: int,
    test_rows: int,
    num_users: int,
    num_items: int,
    seed: int,
) -> list[Path]:
    rng = random.Random(seed)
    return [
        _write_split(output_dir, "train", train_rows, 0, num_users, num_items, False, rng),
        _write_split(output_dir, "valid", valid_rows, 100_000, num_users, num_items, True, rng),
        _write_split(output_dir, "test", test_rows, 200_000, num_users, num_items, True, rng),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw CSV data for SmokeTest.")
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw/SmokeTest"))
    parser.add_argument("--train_rows", type=int, default=96)
    parser.add_argument("--valid_rows", type=int, default=24)
    parser.add_argument("--test_rows", type=int, default=24)
    parser.add_argument("--num_users", type=int, default=12)
    parser.add_argument("--num_items", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    paths = generate_smoke_data(
        output_dir=args.output_dir,
        train_rows=args.train_rows,
        valid_rows=args.valid_rows,
        test_rows=args.test_rows,
        num_users=args.num_users,
        num_items=args.num_items,
        seed=args.seed,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
