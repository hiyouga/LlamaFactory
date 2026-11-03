import pytest

from llamafactory.hparams.data_args import DataArguments


@pytest.mark.parametrize("value", ["", " ", ","])
def test_empty_dataset_string_becomes_none(value):
    args = DataArguments(dataset=value)

    assert args.dataset is None


def test_empty_dataset_with_val_size_is_rejected():
    with pytest.raises(ValueError, match="Cannot specify `val_size`"):
        DataArguments(dataset="", val_size=0.1)


def test_normal_dataset_list_is_untouched():
    args = DataArguments(dataset="alpaca, sharegpt")

    assert args.dataset == ["alpaca", "sharegpt"]
