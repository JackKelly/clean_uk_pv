"""Script to download a small subset of ARCO-ERA5 from Google Cloud, and store
locally as Parquet.

This script only downloads the ERA5 grid cells which cover solar PV systems in
the uk_pv dataset.
"""

import xarray as xr
from clean_uk_pv import geospatial
import polars as pl
from datetime import datetime


def open_arco_era5(
    url: str = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
) -> xr.Dataset:
    """Lazily open the ARCO-ERA5 dataset from Google Cloud Storage."""
    return xr.open_zarr(url, chunks=None, storage_options=dict(token="anon"))


def select_great_britain(ds: xr.Dataset) -> xr.Dataset:
    """Select the Great Britain region from the ARCO-ERA5 dataset."""
    gb_bounding_box = geospatial.GBBoundingBox()
    longitudes_360 = gb_bounding_box.filter_longitudes_360(ds.longitude.values)
    return ds.sel(
        latitude=gb_bounding_box.north_south_slice(),
        longitude=longitudes_360,
    )


def check_longitude_values_are_in_the_range_0_to_360(df: pl.LazyFrame | pl.DataFrame) -> None:
    """Check that the longitude values are in the range [0, 360).

    Note that this function will load all the longitude values, so it will be
    slow for large dataframes.
    """
    min_longitude = df.select("longitude").min()
    max_longitude = df.select("longitude").max()
    if isinstance(df, pl.LazyFrame):
        min_longitude, max_longitude = pl.collect_all(
            [min_longitude, max_longitude], engine="streaming"
        )
        min_longitude = min_longitude.item()
        max_longitude = max_longitude.item()

    if min_longitude < 0 or max_longitude > 360:
        raise ValueError(
            f"Longitude values must be in the range [0, 360). "
            f"Found min: {min_longitude}, max: {max_longitude}"
        )


def convert_longitude_from_360_to_plus_minus_180(
    df: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame | pl.DataFrame:
    """Convert longitude from [0, 360) to [-180, +180] range."""
    return df.with_columns(
        longitude=pl.when(pl.col.longitude > 180)
        .then(pl.col.longitude - 360)
        .otherwise(pl.col.longitude)
    )


def create_spatial_index_from_arco_era5_dataset(ds: xr.Dataset) -> pl.DataFrame:
    """Create discrete spatial index from ARCO-ERA5, so we can join this
    spatial index to the rest of the ARCO-ERA5 data we download.

    Returns a pl.DataFrame with columns: latitude, longitude, and spatial_index.
    """
    df = pl.from_pandas(ds.sel(time=datetime(2025, 3, 1, 0, 0)).to_dataframe().reset_index())
    df = df.select(["latitude", "longitude"])
    check_longitude_values_are_in_the_range_0_to_360(df)
    df = convert_longitude_from_360_to_plus_minus_180(df)
    # Sort to achieve row-major order, starting from the top left (north west).
    df = df.sort(["latitude", "longitude"], descending=[True, False])
    df = df.with_row_index("spatial_index").cast({"spatial_index": pl.UInt16})
    sanity_check_spatial_indicies(df)
    return df


def sanity_check_spatial_indicies(spatial_index_df: pl.DataFrame) -> None:
    """Sanity check our spatial indexes:"""

    from polars import testing as pl_testing

    gb_bounding_box = geospatial.GBBoundingBox()
    gb_spatial_index = geospatial.SpatialIndex(gb_bounding_box)
    assert len(spatial_index_df) == len(gb_spatial_index)
    computed_spatial_index = (
        spatial_index_df.select(["latitude", "longitude"])
        .map_rows(lambda row: gb_spatial_index.lat_lon_to_index(lat=row[0], lon=row[1]))
        .to_series()
        .rename("spatial_index")
        .cast(pl.UInt16)
    )

    pl_testing.assert_series_equal(computed_spatial_index, spatial_index_df["spatial_index"])
