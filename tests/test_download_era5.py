from clean_uk_pv import download_era5
import polars as pl
import pytest


def test_check_longitude_values_are_in_the_range_0_to_360():
    """Test that the longitude values are in the range [0, 360)."""
    df = pl.DataFrame({"longitude": [0, 180, 359]})
    download_era5.check_longitude_values_are_in_the_range_0_to_360(df)
    download_era5.check_longitude_values_are_in_the_range_0_to_360(df.lazy())


@pytest.mark.parametrize(
    "longitude_values",
    [
        [-1, 0, 180, 359],
        [0, 180, 360],
        [0, 180, 361],
        [0, 180, -1],
        [0, 180, -2000],
    ],
)
def test_check_longitude_values_are_in_the_range_0_to_360_invalid(longitude_values: list[float]):
    df = pl.DataFrame({"longitude": longitude_values})
    with pytest.raises(ValueError):
        download_era5.check_longitude_values_are_in_the_range_0_to_360(df)

    with pytest.raises(ValueError):
        download_era5.check_longitude_values_are_in_the_range_0_to_360(df.lazy())


@pytest.mark.parametrize(
    ("longitude_360", "longitude_180"),
    [
        ([0, 0], [0, 0]),
        ([0, 180], [0, 180]),
        ([180, 359], [180, -1]),
        ([0, 180, 270], [0, 180, -90]),
    ],
)
def test_convert_longitude_from_360_to_plus_minus_180(
    longitude_360: list[float], longitude_180: list[float]
):
    """Test that the longitude values are converted from [0, 360) to [-180, +180]."""
    df = pl.DataFrame({"longitude": longitude_360})
    converted_df = download_era5.convert_longitude_from_360_to_plus_minus_180(df)
    assert converted_df["longitude"].to_list() == longitude_180
    converted_df = download_era5.convert_longitude_from_360_to_plus_minus_180(df.lazy())
    assert converted_df.collect()["longitude"].to_list() == longitude_180
