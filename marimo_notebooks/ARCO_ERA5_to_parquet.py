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
    import clean_uk_pv
    return datetime, mo, np, pl, pytest, xr


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
def _(ds):
    GB_LATITUDE = slice(59.5, 49.5)
    GB_LONGITUDE_EAST = 2
    GB_LONGITUDE_WEST = -8

    # ARCO-ERA5 represents longitude as degrees east of Greenwich.
    # So, in ERA5, the westerly point of GB is encoded as 360 degrees - 8 degrees.
    # So the ugly line below appends the longitudes a little below 360 with those a little above 0.
    GB_LONGITUDE = list(
        filter(lambda longitude: longitude > 360 + GB_LONGITUDE_WEST, ds.longitude.values)
    ) + list(filter(lambda longitude: longitude < GB_LONGITUDE_EAST, ds.longitude.values))
    return GB_LATITUDE, GB_LONGITUDE, GB_LONGITUDE_EAST, GB_LONGITUDE_WEST


@app.cell
def _(GB_LATITUDE, GB_LONGITUDE_EAST, GB_LONGITUDE_WEST, np):
    def lat_lon_to_discrete_index(lat: float, lon: float) -> np.uint16:
        # Sanity-check `lat`:
        min_lat = min(GB_LATITUDE.start, GB_LATITUDE.stop)
        max_lat = max(GB_LATITUDE.start, GB_LATITUDE.stop)
        if not min_lat <= lat <= max_lat:
            raise ValueError(f"`lat` must be between {min_lat} and {max_lat}, not {lat}.")

        # Sanity-check `lon`:
        min_lon = min(GB_LONGITUDE_EAST, GB_LONGITUDE_WEST)
        max_lon = max(GB_LONGITUDE_EAST, GB_LONGITUDE_WEST)
        if not min_lon <= lon <= max_lon:
            raise ValueError(f"`lon` must be between {min_lon} and {max_lon}, not {lon}.")

    return (lat_lon_to_discrete_index,)


@app.cell
def collection_of_tests_for_lat_lon_to_discrete_index(
    lat_lon_to_discrete_index,
    pytest,
):
    @pytest.mark.parametrize(("lat", "lon"), [(1000, 0), (0, 1000)])
    def test_lat_lon_to_discrete_index_out_of_bounds(lat, lon):
        with pytest.raises(ValueError):
            lat_lon_to_discrete_index(lat, lon)

    return


@app.cell
def _(GB_LATITUDE, GB_LONGITUDE, datetime, ds):
    xr_ds = (
        ds["2m_temperature"]
        .sel(
            latitude=GB_LATITUDE,
            longitude=GB_LONGITUDE,
            time=slice(datetime(2025, 3, 1, 0, 0), datetime(2025, 3, 2, 0, 0)),
        )
        .load()
    )

    xr_ds
    return (xr_ds,)


@app.cell
def _(pl, xr_ds):
    pl.from_pandas(xr_ds.to_dataframe().reset_index()).unique(["latitude", "longitude"])
    return


@app.cell
def _(pl, plh3, xr_ds):
    df = (
        pl.from_pandas(xr_ds.to_dataframe().reset_index())
        .with_columns(
            h3=plh3.latlng_to_cell("latitude", "longitude", resolution=6, return_dtype=pl.UInt64)
        )
        .drop(["longitude", "latitude"])
        .sort(by=["time", "h3"])
    )

    df.unique("h3")
    return (df,)


@app.cell
def _(df, plh3):
    df.with_columns(h3_res_5=plh3.cell_to_parent("h3", 5))
    return


@app.cell
def _(datetime, df, pl, plh3):
    plh3.graphing.plot_hex_fills(
        df.with_columns(h3_res_4=plh3.cell_to_parent("h3", 4)).filter(
            pl.col.time == datetime(2025, 3, 1, 0, 0)
        ),
        "h3_res_4",
        "2m_temperature",
    )
    return


@app.cell
def _(df, pl, pv_locations):
    df_filtered = df.filter(pl.col.h3.is_in(pv_locations))

    df_filtered
    return (df_filtered,)


@app.cell
def _(df_filtered):
    df_filtered["h3"].unique()
    return


@app.cell
def _(datetime, df_filtered, pl, plh3):
    plh3.graphing.plot_hex_fills(
        df_filtered.filter(pl.col.time == datetime(2025, 3, 1, 0, 0)),
        "h3",
        "2m_temperature",
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


if __name__ == "__main__":
    app.run()
