import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def package_training(root, data_dir, destination):
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "frozen":
        raise ValueError("Freeze the reviewed corpus before packaging it.")
    for name in ["train.jsonl", "validation.jsonl"]:
        if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError(f"Frozen file changed: {name}")
    files = [(data_dir / name, "data/" + name) for name in ["train.jsonl", "validation.jsonl", "manifest.json"]]
    files += [(root / "models" / name, name) for name in ["qwen_train_lora.ipynb", "README.md",
        "requirements-training.in", "requirements-training.lock", "requirements-export.in", "requirements-export.lock"]]
    checksums = {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in files}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path, name in files:
            bundle.write(path, name)
        bundle.writestr("package_checksums.json", json.dumps(checksums, indent=2))
    with zipfile.ZipFile(temporary) as bundle:
        for name, expected in checksums.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                raise ValueError(f"Packaged file changed: {name}")
    temporary.replace(destination)
    return destination


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Package a frozen financial-analysis corpus and its training notebook.")
    parser.add_argument("--data-dir", type=Path, default=root / "workspace/financial_analyst/training/frozen")
    parser.add_argument("--output", type=Path, default=root / "workspace/financial_analyst/training_bundle.zip")
    args = parser.parse_args()
    print(package_training(root, args.data_dir, args.output))
