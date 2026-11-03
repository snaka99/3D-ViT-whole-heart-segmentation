# 3D CT and MRI Segmentation with ViT

Python refactor of my research notebooks for 3D medical image segmentation using PyTorch and MONAI. The CT and MRI experiments use the same model and training code, with separate settings.

## Project structure

```text
configs/               CT and MRI settings
src/medical_vit/       Data loading, transforms, model, training, metrics, inference
tests/                 Basic checks
main.py                Command-line entry point
requirements.txt       Dependencies
```

## Setup

Requires Python 3.10+ and a PyTorch installation suitable for your system.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

## Training

Put the original MONAI Decathlon-style dataset lists and referenced NIfTI volumes in `data/` (not included). By default, CT expects `data/dataset_1.json` and MRI expects `data/dataset_0.json`. The JSON must include a `training` split with `image` and `label` keys.

```bash
python main.py --config configs/ct.yaml train
python main.py --config configs/mri.yaml train
```

To run a short technical check:

```bash
python main.py --config configs/ct.yaml train --max-iterations 10 --eval-interval 5
```

Other commands:

```bash
python main.py --config configs/ct.yaml plot --metrics results/ct/summary.json --output results/ct/plot.png
python main.py --config configs/ct.yaml predict --checkpoint results/ct/fold_1/best_model.pth --image data/example.nii.gz --output results/ct/example_pred.nii.gz
```

These examples assume the corresponding input files exist. Do not commit patient images, original datasets, or model weights without the required permissions.


## Labeled test evaluation

Test labels are used **only for evaluation**, never for training, fold selection or tuning.
Keep the test volumes outside Git. Give the image and label the same filename:

```text
data/test/images/case_001.nii.gz
data/test/labels/case_001.nii.gz
```

Evaluate a trained fold checkpoint:

```powershell
python main.py --config configs/ct.yaml --device cpu evaluate --checkpoint results/ct/fold_1/best_model.pth --images-dir data/test/images --labels-dir data/test/labels --output results/ct/test_evaluation
```

For MRI, replace the config and checkpoint. Alternatively, provide a Decathlon-style
JSON with a `testing` split containing `image` and `label` fields:

```powershell
python main.py --config configs/ct.yaml evaluate --checkpoint results/ct/fold_1/best_model.pth --dataset data/dataset_1.json --output results/ct/test_evaluation
```

Outputs: `test_metrics.csv` (per-case and per-class Dice, IoU and HD95) and
`test_summary.json` (aggregate case metrics). Results are evaluated in the
**resized/preprocessed image space**, not the original image grid. HD95 is
therefore reported in **processed voxels, not millimeters**. Empty classes
are excluded from macro averages; HD95 is undefined when only one mask is empty.

Do not compare these processed-grid metrics directly to results calculated
in the original image space. Do not evaluate with untrained model weights.
If you compare all five fold checkpoints on the test set, report them as five
models, not as five independent test sets.

## Notes

- Five-fold cross-validation and the original modality-specific image dimensions are retained.
- Augmentation is limited to the training cases of each fold to prevent validation leakage. This changes the experimental procedure compared with the notebooks and may change Dice scores.
- A separate best and last checkpoint is saved for each fold.
- Inference uses deterministic preprocessing. This differs from the notebooks' stochastic transforms for some unlabeled examples.
- The code has not been retrained on the original CT/MRI images. The published thesis results should not be described as results of this refactored pipeline without re-evaluation.

The detailed comparison with the notebooks is in [migration notes](docs/MIGRATION_NOTES.md).

## Official MMWHS evaluation (encrypted test labels)

The official MMWHS test labels are encrypted. The normal Python `evaluate` command **cannot** read them as plain masks. Use the dataset's published Windows tools (`zxhtransform.exe` and `zxhCardWhsEvaluate.exe`) to evaluate predictions, reproducing the commands in `matlab_evaluation_script.m`. `zxhimageop.exe` can be used separately to inspect image metadata.

Example local layout (not committed):

```text
data/mmwhs/labels/ct_test_2001_label_encrypt_1mm.nii.gz
data/mmwhs/predictions/ct/ct_test_2001_label.nii.gz
tools/mmwhs/zxhtransform.exe
tools/mmwhs/zxhCardWhsEvaluate.exe
```

Check the command without running the evaluators:

```powershell
python main.py --config configs/ct.yaml evaluate-official --predictions-dir data/mmwhs/predictions/ct --labels-dir data/mmwhs/labels --tools-dir tools/mmwhs --output results/mmwhs_ct --dry-run
```

Then remove `--dry-run` to run official evaluation. For MRI, use `configs/mri.yaml` and MR predictions (`mr_test_2001_label.nii.gz`).

**Important:** Prediction masks must match the *original* CT/MR physical image space (correct affine, orientation, shape, spacing, and integer label values). The existing `predict` function saves masks in resized preprocessing space and is **not automatically suitable** for official evaluation. Restore predictions to original image space and visually verify alignment before official scoring. Resampling to 1 mm alone does not repair an incorrect affine or orientation. This wrapper does **not** perform that restoration or decrypt labels itself.

Outputs use the names from the official MATLAB script (`ct_dice.xls`, `ct_jaccard.xls`, `ct_surface`). They are created by the official executables. `cases.csv` records input pairs; output files may not be ordinary Excel workbooks. Use a new output directory per run. The `.exe` files are not included in this repository.
