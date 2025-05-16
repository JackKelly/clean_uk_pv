import marimo

__generated_with = "0.13.8"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import polars as pl
    from datetime import date, datetime, timedelta
    import xarray as xr
    import marimo as mo
    import pytest
    from clean_uk_pv import geospatial

    return date, datetime, geospatial, pl, timedelta, xr


@app.cell
def _():
    VARIABLES = [
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
    return


@app.cell
def _(xr):
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        chunks=None,
        storage_options=dict(token="anon"),
    )
    return (ds,)


@app.cell
def _(datetime, ds, geospatial):
    gb_bounding_box = geospatial.GBBoundingBox()

    xr_ds = (
        ds["2m_temperature"]
        .sel(
            latitude=gb_bounding_box.north_south_slice(),
            longitude=gb_bounding_box.filter_longitudes_360(ds.longitude.values),
            time=slice(datetime(2025, 3, 1, 0, 0), datetime(2025, 3, 2, 0, 0)),
        )
        .load()
    )

    xr_ds
    return gb_bounding_box, xr_ds


@app.cell
def _(datetime, xr_ds):
    import matplotlib.pyplot as plt

    plt.imshow(xr_ds.sel(time=datetime(2025, 3, 1, 0, 0)).values)
    return


@app.cell
def _(datetime, pl, xr_ds):
    """Get a Polars DataFrame of a single time slice, so we can create a Polars DF of the spatial index"""

    spatial_index_df = (
        pl.from_pandas(xr_ds.sel(time=datetime(2025, 3, 1, 0, 0)).to_dataframe().reset_index())
        .select(["latitude", "longitude"])
        # Convert longitude from ERA5's [0, 360) range, to [-180, +180]:
        .with_columns(
            longitude=pl.when(pl.col.longitude > 180)
            .then(pl.col.longitude - 360)
            .otherwise(pl.col.longitude)
        )
        # Sort to achieve row-major order, starting from the top left (north west).
        .sort(["latitude", "longitude"], descending=[True, False])
        .with_row_index("spatial_index")
        .cast({"spatial_index": pl.UInt16})
    )

    spatial_index_df
    return (spatial_index_df,)


@app.cell
def _(gb_bounding_box, geospatial, pl, spatial_index_df):
    """Sanity check our spatial indexes:"""

    from polars import testing as pl_testing

    gb_spatial_index = geospatial.SpatialIndex(gb_bounding_box)
    assert len(spatial_index_df) == len(gb_spatial_index)
    _computed_spatial_index = (
        spatial_index_df.select(["latitude", "longitude"])
        .map_rows(lambda _row: gb_spatial_index.lat_lon_to_index(lat=_row[0], lon=_row[1]))
        .to_series()
        .rename("spatial_index")
        .cast(pl.UInt16)
    )

    pl_testing.assert_series_equal(_computed_spatial_index, spatial_index_df["spatial_index"])
    return (gb_spatial_index,)


@app.cell
def _(gb_spatial_index, pl):
    """Get spatial indexes for PV systems."""

    pv_systems = pl.read_csv("~/data/uk_pv/metadata.csv", try_parse_dates=True)
    pv_systems

    pv_system_spatial_index = (
        pv_systems.select(["latitude_rounded", "longitude_rounded"])
        .map_rows(lambda _row: gb_spatial_index.lat_lon_to_index(lat=_row[0], lon=_row[1]))
        .to_series()
        .rename("spatial_index")
        .cast(pl.UInt16)
        .unique()
        .sort()
        .implode()
    )

    pv_system_spatial_index
    return pv_system_spatial_index, pv_systems


@app.cell
def _(pv_systems):
    pv_systems
    return


@app.cell
def _(
    date,
    ds,
    gb_bounding_box,
    pl,
    pv_system_spatial_index,
    spatial_index_df,
    timedelta,
):
    import pathlib

    def load_month_of_era5_and_save_parquet(
        start_date: date,
        variable_name: str,
        output_path_root: pathlib.Path,
        overwrite: bool = False,
    ):
        # Check if output file already exists:
        full_output_path = (
            output_path_root
            / variable_name
            / f"year={start_date.year}/month={start_date.month:02d}"
        )
        if (full_output_path / "data.parquet").exists() and not overwrite:
            return

        # Load ERA5 from Zarr on Google Cloud:
        end_datetime = (start_date + timedelta(days=40)).replace(day=1) - timedelta(hours=1)
        xr_ds = (
            ds[variable_name]
            .sel(
                latitude=gb_bounding_box.north_south_slice(),
                longitude=gb_bounding_box.filter_longitudes_360(ds.longitude.values),
                time=slice(start_date, end_datetime),
            )
            .load()
        )

        # Convert to Polars DataFrame:
        df = pl.from_pandas(xr_ds.to_dataframe().reset_index())
        del xr_ds
        df = (
            df
            # Convert longitude from ERA5's [0, 360) range, to [-180, +180]:
            .with_columns(
                longitude=pl.when(pl.col.longitude > 180)
                .then(pl.col.longitude - 360)
                .otherwise(pl.col.longitude)
            )
            .join(spatial_index_df, on=["latitude", "longitude"], how="left")
            .drop(["latitude", "longitude"])
            .filter(pl.col.spatial_index.is_in(pv_system_spatial_index))
            .sort(["spatial_index", "time"])
        )

        full_output_path.mkdir(exist_ok=True, parents=True)
        df.write_parquet(full_output_path / "data.parquet", statistics="full")

    # load_month_of_era5_and_save_parquet(
    #    datetime(2025, 3, 1, 0, 0),
    #    "surface_solar_radiation_downwards",
    #    pathlib.Path("~/data/ERA5/").expanduser(),
    # )
    return load_month_of_era5_and_save_parquet, pathlib


@app.cell
def _(load_month_of_era5_and_save_parquet, pathlib, pl, pv_systems):
    """Load data month-by-month"""

    from concurrent import futures
    import functools

    start_date = pv_systems["start_datetime_GMT"].min().date().replace(day=1)
    end_date = pv_systems["end_datetime_GMT"].max().date()

    months = pl.date_range(start_date, end_date, interval="1mo", eager=True)

    VARIABLES_SELECTED = [
        # "10m_u_component_of_wind",
        # "10m_v_component_of_wind",
        # "2m_temperature",
        # "clear_sky_direct_solar_radiation_at_surface",
        # "mean_surface_direct_short_wave_radiation_flux",
        "surface_thermal_radiation_downwards",
        "total_cloud_cover",
    ]

    for variable_name in VARIABLES_SELECTED:
        print(variable_name)
        with futures.ThreadPoolExecutor(max_workers=4) as e:
            results = e.map(
                functools.partial(
                    load_month_of_era5_and_save_parquet,
                    variable_name=variable_name,
                    output_path_root=pathlib.Path("~/data/ERA5/").expanduser(),
                ),
                months,
            )

            # Surface any errors raised during concurrent execution:
            list(results)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
