import csv
import re
import subprocess
from pathlib import Path


CASE_PATTERN = re.compile(r"^(ct|mr)_test_(20(?:0[1-9]|[1-3][0-9]|40))(?:_label)?\.nii\.gz$", re.IGNORECASE)


def find_predictions(predictions_dir: Path, modality: str) -> list[tuple[str, Path]]:
    found = {}
    for path in sorted(predictions_dir.glob("*.nii.gz")):
        match = CASE_PATTERN.fullmatch(path.name)
        if match and match.group(1).lower() == modality:
            case_id = match.group(2)
            if case_id in found:
                raise ValueError(f"Multiple predictions for {modality} case {case_id}")
            found[case_id] = path
    if not found:
        raise FileNotFoundError(f"No {modality.upper()} MMWHS predictions in {predictions_dir}")
    return sorted(found.items())


def evaluate_official(modality: str, predictions_dir: Path, labels_dir: Path,
                      tools_dir: Path, output_dir: Path, dry_run: bool = False) -> Path:
    modality = "mr" if modality == "mri" else modality
    if modality not in ("ct", "mr"):
        raise ValueError("Expected CT or MRI modality")
    predictions_dir = Path(predictions_dir)
    labels_dir = Path(labels_dir)
    tools_dir = Path(tools_dir)
    output_dir = Path(output_dir)
    cases = find_predictions(predictions_dir, modality)
    transform = tools_dir / "zxhtransform.exe"
    evaluator = tools_dir / "zxhCardWhsEvaluate.exe"
    for tool in (transform, evaluator):
        if not tool.is_file():
            raise FileNotFoundError(f"Missing official MMWHS tool: {tool}")
    for case_id, _ in cases:
        label = labels_dir / f"{modality}_test_{case_id}_label_encrypt_1mm.nii.gz"
        if not label.is_file():
            raise FileNotFoundError(f"Missing encrypted label: {label}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "dice": output_dir / f"{modality}_dice.xls",
        "jaccard": output_dir / f"{modality}_jaccard.xls",
        "surface": output_dir / f"{modality}_surface",
    }
    for path in outputs.values():
        if path.exists() and not dry_run:
            raise FileExistsError(f"Output already exists; use a new output directory: {path}")
    manifest = output_dir / "cases.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "prediction", "encrypted_label"])
        for case_id, prediction in cases:
            label = labels_dir / f"{modality}_test_{case_id}_label_encrypt_1mm.nii.gz"
            writer.writerow([modality + case_id, str(prediction), str(label)])

    for case_id, prediction in cases:
        label = labels_dir / f"{modality}_test_{case_id}_label_encrypt_1mm.nii.gz"
        resampled = output_dir / f"{modality}_test_{case_id}_1mm.nii.gz"
        commands = [
            [str(transform), str(prediction), "-o", str(resampled), "-resave",
             "-spacing", "1", "1", "1", "-nearest", "-v", "0"],
            *([
                str(evaluator), str(resampled), str(label), metric, "ALL", str(destination),
                f"{modality}{case_id}", "--decodeseg2",
            ] for metric, destination in [
                ("0", outputs["dice"]), ("1", outputs["jaccard"]),
                ("surface", outputs["surface"]),
            ]),
        ]
        for command in commands:
            if dry_run:
                print(subprocess.list2cmdline(command))
            else:
                subprocess.run(command, check=True)
        print(f"{modality.upper()} {case_id}: {'commands shown' if dry_run else 'evaluated'}")
    return manifest
