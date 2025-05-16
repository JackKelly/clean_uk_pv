"""Script to download a small subset of ARCO-ERA5 from Google Cloud, and store
locally as Parquet.

This script only downloads the ERA5 grid cells which cover solar PV systems in
the uk_pv dataset.
"""

import xarray as xr
from clean_uk_pv import geospatial
import polars as pl
from datetime import date, datetime, timedelta
import pathlib
from concurrent import futures
import functools
import click
from collections.abc import Iterable


ALL_RELEVANT_ERA5_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",  # Not sure if we need this for solar???
    "2m_temperature",
    "clear_sky_direct_solar_radiation_at_surface",
    "fraction_of_cloud_cover",
    "high_cloud_cover",
    "low_cloud_cover",
    "mean_surface_direct_short_wave_radiation_flux",
    "mean_total_precipitation_rate",
    "medium_cloud_cover",
    "snow_depth",
    "surface_pressure",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "total_cloud_cover",
    "total_sky_direct_solar_radiation_at_surface",
]


def open_arco_era5(
    url: str = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
) -> xr.Dataset:
    """Lazily open the ARCO-ERA5 dataset from Google Cloud Storage."""
    return xr.open_zarr(url, chunks=None, storage_options=dict(token="anon"))


def select_great_britain(ds: xr.Dataset) -> xr.Dataset:
    """Select the Great Britain region from the ARCO-ERA5 dataset."""
    gb_bounding_box = geospatial.GBBoundingBox()
    gb_longitudes_360 = gb_bounding_box.filter_longitudes_360(ds.longitude.values)
    return ds.sel(
        latitude=gb_bounding_box.north_south_slice(),
        longitude=gb_longitudes_360,
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

    if min_longitude < 0 or max_longitude >= 360:
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
    # Select an arbitrary ERA5 variable and datetime: We're just interested in the lat and lon.
    df = pl.from_pandas(
        ds["2m_temperature"].sel(time=datetime(2025, 3, 1, 0, 0)).to_dataframe().reset_index()
    )
    df = df.select(["latitude", "longitude"])
    check_longitude_values_are_in_the_range_0_to_360(df)
    df = convert_longitude_from_360_to_plus_minus_180(df)
    # Sort to achieve row-major order, starting from the top left (north west).
    df = df.sort(["latitude", "longitude"], descending=[True, False])
    df = df.with_row_index("spatial_index").cast({"spatial_index": pl.UInt16})
    sanity_check_spatial_indicies(df)
    return df


def get_gb_spatial_index() -> geospatial.SpatialIndex:
    """Get the spatial index for Great Britain."""
    gb_bounding_box = geospatial.GBBoundingBox()
    return geospatial.SpatialIndex(gb_bounding_box)


def sanity_check_spatial_indicies(spatial_index_df: pl.DataFrame) -> None:
    """Sanity check our spatial indexes:"""

    from polars import testing as pl_testing

    gb_spatial_index = get_gb_spatial_index()
    assert len(spatial_index_df) == len(gb_spatial_index)
    computed_spatial_index = (
        spatial_index_df.select(["latitude", "longitude"])
        .map_rows(lambda row: gb_spatial_index.lat_lon_to_index(lat=row[0], lon=row[1]))
        .to_series()
        .rename("spatial_index")
        .cast(pl.UInt16)
    )

    pl_testing.assert_series_equal(computed_spatial_index, spatial_index_df["spatial_index"])


def load_spatial_index_for_uk_pv_systems(metadata_filename: str) -> pl.Array:
    """Get spatial indexes for PV systems in UK PV dataset."""

    pv_systems = pl.read_csv(metadata_filename)

    gb_spatial_index = get_gb_spatial_index()
    return (
        pv_systems.select(["latitude_rounded", "longitude_rounded"])
        .map_rows(lambda _row: gb_spatial_index.lat_lon_to_index(lat=_row[0], lon=_row[1]))
        .to_series()
        .rename("spatial_index")
        .cast(pl.UInt16)
        .unique()
        .sort()
        .implode()
    )


def add_month(dt: datetime) -> datetime:
    """Add one month to a datetime object, handling year overflow."""
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1)
    else:
        return dt.replace(month=dt.month + 1)


def load_month_of_era5_and_save_parquet(
    start_date: date,
    ds: xr.Dataset,
    variable_name: str,
    output_path_root: pathlib.Path,
    spatial_index_with_lat_lon: pl.DataFrame,
    overwrite: bool = False,
) -> None:
    """
    Load a month of ERA5 data from the ARCO-ERA5 dataset, and save it as Parquet.

    Parameters:
    - start_date: datetime.date
        The first day of the month to download.
    - ds: xarray.Dataset
        The lazyily opened ARCO-ERA5 dataset. This must already have been filtered
        to include only Great Britain.
    - variable_name: str
        The name of the variable to download (e.g. "10m_u_component_of_wind",
        "10m_v_component_of_wind", "surface_pressure", "total_cloud_cover").
    - output_path_root: pathlib.Path
        The root path to save the Parquet files to. The files will be saved in
        a subdirectory with the name of the variable and the year and month of
        the start date.
    - spatial_index_with_lat_lon: pl.DataFrame
        A DataFrame with columns spatial_index, latitude and longitude. This should already
        be filtered to only include values for the PV systems in the UK PV dataset,
        if that is the desired behaviour.
    - overwrite: bool
        If True, overwrite the output file if it already exists. If False, skip
        downloading if the output file already exists.
    """
    # Check if output file already exists:
    full_output_path = (
        output_path_root
        / f"variable={variable_name}"
        / f"year={start_date.year}"
        / f"month={start_date.month:02d}"
    )
    full_output_filename = full_output_path / "data.parquet"
    if full_output_filename.exists() and not overwrite:
        print(f"Output file {full_output_filename} already exists. Skipping download.")
        return
    else:
        print(f"Downloading month of data starting on {start_date} for {variable_name}...")

    # Load ERA5 from Zarr on Google Cloud:
    assert start_date.day == 1, "start_date must be the first day of the month"
    # Subtract 1 hour from the end date to get the last hour of the month.
    # This is necessary because `xarray.sel(time=slice(start, end))` is *inclusive* of `start` and `end`.
    end_datetime = add_month(start_date) - timedelta(hours=1)
    ds_filtered = ds[variable_name].sel(time=slice(start_date, end_datetime)).load()

    # Convert to Polars DataFrame:
    df = pl.from_pandas(ds_filtered.to_dataframe().reset_index())
    del ds_filtered
    df = convert_longitude_from_360_to_plus_minus_180(df)
    df = (
        df.join(spatial_index_with_lat_lon, on=["latitude", "longitude"], how="semi")
        .drop(["latitude", "longitude"])
        .sort(["spatial_index", "time"])
    )

    print(f"Writing {full_output_filename}...")
    full_output_path.mkdir(exist_ok=True, parents=True)
    df.write_parquet(full_output_filename, statistics="full")


def get_start_and_end_dates_from_uk_pv_metadata(metadata_filename: str) -> [date, date]:
    pv_systems = pl.read_csv(metadata_filename, try_parse_dates=True)
    start_date = pv_systems["start_datetime_GMT"].min().date().replace(day=1)
    end_date = pv_systems["end_datetime_GMT"].max().date()
    return [start_date, end_date]


def get_spatial_index_with_lat_lon_filtered_by_uk_pv_systems(
    ds: xr.Dataset, metadata_filename: str
) -> pl.DataFrame:
    """Get the spatial index with latitude and longitude, filtered by the UK PV systems."""
    spatial_index_with_lat_lon = create_spatial_index_from_arco_era5_dataset(ds)
    spatial_index_for_uk_pv_systems = load_spatial_index_for_uk_pv_systems(metadata_filename)
    return spatial_index_with_lat_lon.filter(
        pl.col.spatial_index.is_in(spatial_index_for_uk_pv_systems)
    )


def _process_list(_ctx: click.Context, _param: click.Parameter, value) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",")]


@click.command(help=__doc__)
@click.option(
    "--start_date",
    type=click.DateTime(),
    default=None,
    help=(
        "Optional. The start date of the data to download."
        " Must be the first day of the month. Defaults to the first month of UK PV data."
    ),
)
@click.option(
    "--end_date",
    type=click.DateTime(),
    default=None,
    help=(
        "Optional. The end date of the data to download. Defaults to the last month of UK PV data."
    ),
)
@click.option("--metadata_filename", type=click.Path(dir_okay=False, exists=True), required=True)
@click.option("--parquet_output_path", type=click.Path(file_okay=False), required=True)
@click.option("--overwrite", flag_value=True)
@click.option(
    "--era5_variable_names",
    type=click.STRING,
    callback=_process_list,
    default=ALL_RELEVANT_ERA5_VARIABLES,
    help=(
        "A list of ERA5 variable names to download, specifed as a comma-separated list."
        f" Defaults to:\n\n{','.join(ALL_RELEVANT_ERA5_VARIABLES)}"
    ),
)
def main(
    start_date: date | None,
    end_date: date | None,
    metadata_filename: pathlib.Path,
    parquet_output_path: pathlib.Path,
    overwrite: bool,
    era5_variable_names: Iterable[str],
) -> None:
    if start_date is None or end_date is None:
        uk_pv_start, uk_pv_end = get_start_and_end_dates_from_uk_pv_metadata(metadata_filename)
        start_date = start_date or uk_pv_start
        end_date = end_date or uk_pv_end

    print(f"start_date = {start_date}")
    print(f"end_date = {end_date}")
    print(f"metadata_filename = {metadata_filename}")
    print(f"parquet_output_path = {parquet_output_path}")
    print(f"overwrite = {overwrite}")
    print(f"era5_variable_names = {era5_variable_names}")

    months = pl.date_range(start_date, end_date, interval="1mo", eager=True)

    print("Opening ARCO-ERA5...")
    ds = open_arco_era5()
    ds = select_great_britain(ds)
    spatial_index_with_lat_lon = get_spatial_index_with_lat_lon_filtered_by_uk_pv_systems(
        ds, metadata_filename
    )

    for variable_name in era5_variable_names:
        print(f"Loading {variable_name}...")
        with futures.ThreadPoolExecutor(max_workers=4) as e:
            results = e.map(
                functools.partial(
                    load_month_of_era5_and_save_parquet,
                    ds=ds,
                    variable_name=variable_name,
                    output_path_root=pathlib.Path(parquet_output_path).expanduser(),
                    spatial_index_with_lat_lon=spatial_index_with_lat_lon,
                    overwrite=overwrite,
                ),
                months,
            )

            # Surface any errors raised during concurrent execution:
            list(results)
