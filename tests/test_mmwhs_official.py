from pathlib import Path
from unittest.mock import patch

import pytest

from medical_vit.mmwhs_official import evaluate_official, find_predictions


def setup_cases(tmp_path):
    pred, labels, tools = (tmp_path / x for x in ("pred", "labels", "tools"))
    for folder in (pred, labels, tools):
        folder.mkdir()
    (pred / "ct_test_2001_label.nii.gz").touch()
    (labels / "ct_test_2001_label_encrypt_1mm.nii.gz").touch()
    (tools / "zxhtransform.exe").touch()
    (tools / "zxhCardWhsEvaluate.exe").touch()
    return pred, labels, tools


def test_case_matches_original_convention(tmp_path):
    pred, _, _ = setup_cases(tmp_path)
    assert find_predictions(pred, "ct")[0][0] == "2001"


def test_missing_encrypted_labels_fails(tmp_path):
    pred, labels, tools = setup_cases(tmp_path)
    (labels / "ct_test_2001_label_encrypt_1mm.nii.gz").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_official("ct", pred, labels, tools, tmp_path / "out", dry_run=True)


def test_command_arguments_and_order(tmp_path):
    pred, labels, tools = setup_cases(tmp_path)
    with patch("medical_vit.mmwhs_official.subprocess.run") as run:
        manifest = evaluate_official("ct", pred, labels, tools, tmp_path / "out")
    assert manifest.exists()
    assert run.call_count == 4
    commands = [call.args[0] for call in run.call_args_list]
    assert commands[0][-8:] == ["-resave", "-spacing", "1", "1", "1", "-nearest", "-v", "0"]
    assert [cmd[3] for cmd in commands[1:]] == ["0", "1", "surface"]
    assert all(cmd[-1] == "--decodeseg2" for cmd in commands[1:])
    assert all(call.kwargs["check"] for call in run.call_args_list)


def test_dry_run_never_invokes_executables(tmp_path):
    pred, labels, tools = setup_cases(tmp_path)
    with patch("medical_vit.mmwhs_official.subprocess.run") as run:
        evaluate_official("ct", pred, labels, tools, tmp_path / "out", dry_run=True)
    run.assert_not_called()
