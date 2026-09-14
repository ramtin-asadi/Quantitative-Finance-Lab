import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantfinlab.analyst.config import AnalystConfig
from quantfinlab.analyst.corpus import build_candidates
from quantfinlab.analyst.documents import DocumentStore
from quantfinlab.analyst.training import chronological_split, freeze_dataset, read_examples


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate the Project 24 SFT corpus.")
    parser.add_argument("stage", choices=["anchors", "candidates", "freeze"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--validation-start", default="2025-01-01T00:00:00Z")
    args = parser.parse_args()
    config = AnalystConfig.from_repo(args.root)
    path = config.workspace / "training"
    if args.stage == "candidates":
        anchors = read_examples(path / "anchors_reviewed.jsonl")
        if len(anchors) < 100 or any(row.quality_status != "accepted" or not row.review for row in anchors):
            raise ValueError("Review and accept at least 100 anchors before scaling.")
    if args.stage in {"anchors", "candidates"}:
        summary = build_candidates(config.root, path, anchors_only=args.stage == "anchors")
        print(json.dumps({key:value for key,value in summary.items() if key != "errors"},indent=2))
    else:
        examples = read_examples(path / "accepted.jsonl")
        train, validation, embargo = chronological_split(examples, validation_start=args.validation_start)
        manifest = freeze_dataset(train, validation, path / "frozen", store=DocumentStore(config.workspace / "documents"))
        print(json.dumps({**manifest,"embargoed_examples":len(embargo)},indent=2))


if __name__ == "__main__":
    main()
