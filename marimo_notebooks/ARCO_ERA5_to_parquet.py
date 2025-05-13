import marimo

__generated_with = "0.13.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import xarray as xr
    from datetime import datetime
    import matplotlib.pyplot as plt
    import polars as pl
    import polars_h3 as plh3
    import numpy as np
    return datetime, mo, np, pl, plh3, plt, xr


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
def _(datetime, ds, plt):
    UK_LATITUDE = slice(61, 50)
    UK_LONGITUDE_EAST = 2
    UK_LONGITUDE_WEST = -10.5
    # ARCO-ERA5 represents longitude as degrees east of Greenwich.
    # So the westerly point of the UK is actually 360 degrees - 12 degrees.
    # So the ugly line below appends the longitudes in the range [360 - 12, 360) with [0, 4].
    UK_LONGITUDE = list(
        filter(lambda longitude: longitude > 360 + UK_LONGITUDE_WEST, ds.longitude.values)
    ) + list(filter(lambda longitude: longitude < UK_LONGITUDE_EAST, ds.longitude.values))

    xr_land_sea_mask = ds["land_sea_mask"].sel(
        latitude=UK_LATITUDE,
        longitude=UK_LONGITUDE,
        time=datetime(2025, 4, 1, 12, 0),
    )

    plt.imshow(xr_land_sea_mask.values)
    return UK_LATITUDE, UK_LONGITUDE, xr_land_sea_mask


@app.cell
def _(np, pl, xr):
    def xarray_data_array_to_polars_data_frame(da: xr.DataArray) -> pl.DataFrame:
        shape = da.shape
        return pl.DataFrame(
            {
                da.name: da.values.flatten(),
                "longitude": np.tile(da.longitude.values, shape[0]),
                "latitude": np.repeat(da.latitude.values, shape[1]),
            }
        )
    return


@app.cell
def _(pl, plh3, xr_land_sea_mask):
    land_sea_mask = (
        pl.from_pandas(xr_land_sea_mask.to_dataframe().reset_index().drop("time", axis="columns"))
        .filter(pl.col.land_sea_mask > 0.1)
        .with_columns(
            h3=plh3.latlng_to_cell("latitude", "longitude", resolution=5, return_dtype=pl.UInt64)
        )
        .drop(["longitude", "latitude"])
    )

    land_sea_mask
    return (land_sea_mask,)


@app.cell
def _(land_sea_mask, plh3):
    plh3.graphing.plot_hex_fills(land_sea_mask, "h3", "land_sea_mask")
    return


@app.cell
def _(land_sea_mask):
    land_sea_mask_h3_cells = land_sea_mask["h3"].unique().sort().implode()
    return (land_sea_mask_h3_cells,)


@app.cell
def _(UK_LATITUDE, UK_LONGITUDE, datetime, ds):
    xr_ds = (
        ds["2m_temperature"]
        .sel(
            latitude=UK_LATITUDE,
            longitude=UK_LONGITUDE,
            time=slice(datetime(2025, 3, 1, 0, 0), datetime(2025, 3, 2, 0, 0)),
        )
        .load()
    )

    xr_ds
    return (xr_ds,)


@app.cell
def _(land_sea_mask_h3_cells, pl, plh3, xr_ds):
    df = (
        pl.from_pandas(xr_ds.to_dataframe().reset_index())
        .with_columns(
            h3=plh3.latlng_to_cell("latitude", "longitude", resolution=5, return_dtype=pl.UInt64)
        )
        .drop(["longitude", "latitude"])
        .filter(pl.col.h3.is_in(land_sea_mask_h3_cells))
        .sort(by=["time", "h3"])
    )

    df
    return (df,)


@app.cell
def _(datetime, df, pl, plh3):
    plh3.graphing.plot_hex_fills(
        df.filter(pl.col.time == datetime(2025, 3, 1, 0, 0)), "h3", "2m_temperature"
    )
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
