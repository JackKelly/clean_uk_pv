import marimo

__generated_with = "0.13.8"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import polars as pl
    from datetime import datetime
    import xarray as xr
    import marimo as mo
    import pytest
    from clean_uk_pv import geospatial
    return datetime, geospatial, mo, pl, xr


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

    _spatial_index = geospatial.SpatialIndex(gb_bounding_box)
    assert len(spatial_index_df) == len(_spatial_index)
    _computed_spatial_index = (
        spatial_index_df.select(["latitude", "longitude"])
        .map_rows(lambda _row: _spatial_index.lat_lon_to_index(lat=_row[0], lon=_row[1]))
        .to_series()
        .rename("spatial_index")
        .cast(pl.UInt16)
    )

    pl_testing.assert_series_equal(_computed_spatial_index, spatial_index_df["spatial_index"])
    return


@app.cell
def _():
    """Get spatial indexes for PV systems."""
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Next steps:

    - Load a day at a time.
    - Loop round days and NWP variables.
    - Use multiple threads to load the xarray data.
    - Store a monthly Parquet when we've loaded a month.
    """
    )
    return


if __name__ == "__main__":
    app.run()
