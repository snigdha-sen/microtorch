import time

from microtorch.utils.helpers import most_recent_output_file, strip_filename


def test_strip_filename_nii_gz():
    path = "/data/sub-01/anat/image.nii.gz"
    assert strip_filename(path) == "image"


def test_strip_filename_nii():
    path = "/data/sub-01/anat/image.nii"
    assert strip_filename(path) == "image"


def test_strip_filename_other_extension():
    path = "/data/sub-01/anat/image.txt"
    assert strip_filename(path) == "image"


def test_most_recent_output_file_returns_none_when_no_match(tmp_path):
    assert most_recent_output_file(tmp_path, "Ball", "hidden_dropout_mlp") is None


def test_most_recent_output_file_picks_most_recently_modified(tmp_path):
    older = tmp_path / "Ball_hidden_dropout_mlp_param_maps.nii.gz"
    newer = tmp_path / "Ball_run2_hidden_dropout_mlp_2_param_maps.nii.gz"

    older.write_text("old")
    time.sleep(0.01)
    newer.write_text("new")

    result = most_recent_output_file(tmp_path, "Ball", "hidden_dropout_mlp")

    assert result == newer


def test_most_recent_output_file_searches_recursively(tmp_path):
    nested = tmp_path / "2026-01-01" / "12-00-00"
    nested.mkdir(parents=True)
    expected = nested / "Ball_hidden_dropout_mlp_param_maps.nii.gz"
    expected.write_text("data")

    result = most_recent_output_file(tmp_path, "Ball", "hidden_dropout_mlp")

    assert result == expected


def test_most_recent_output_file_does_not_match_other_network_type(tmp_path):
    (tmp_path / "Ball_vae_param_maps.nii.gz").write_text("data")

    result = most_recent_output_file(tmp_path, "Ball", "hidden_dropout_mlp")

    assert result is None
