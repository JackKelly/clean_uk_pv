"""Unit tests for geospatial.py."""

from collections.abc import Iterable
from typing import Final

import numpy as np  # type: ignore
import pytest  # type: ignore

from clean_uk_pv import geospatial  # type: ignore


@pytest.fixture
def spatial_index() -> geospatial.SpatialIndex:
    """Get a spatial index for Great Britain with a resolution of 0.25 degrees."""
    return geospatial.SpatialIndex(geospatial.GBBoundingBox(), 0.25)


@pytest.mark.parametrize(
    ("north", "south", "east", "west"),
    [
        (60, 50, -10, 10),  # east < west.
        (50, 60, 30, 10),  # north < south.
        (50, 60, -170, 170),  # Crosses anti-Meridian.
        (190, 0, 30, 20),  # North is too high.
    ],
)
def test_bounding_box_raises(north: float, south: float, east: float, west: float):
    """Test that the BoundingBox raises a ValueError if the coordinates are not valid."""
    with pytest.raises(ValueError):
        geospatial.BoundingBox(north=north, south=south, east=east, west=west)


def test_init_gb_bouding_box():
    gb_bounding_box = geospatial.GBBoundingBox()
    assert gb_bounding_box.north == 59.625
    assert gb_bounding_box.south == 49.625
    assert gb_bounding_box.east == 2.125
    assert gb_bounding_box.west == -8.125
    assert gb_bounding_box.width_degrees == 10.25
    assert gb_bounding_box.height_degrees == 10.0


def test_init_spatial_index(spatial_index: geospatial.SpatialIndex):
    assert spatial_index.n_rows == 40
    assert spatial_index.n_cols == 41


@pytest.mark.parametrize(
    ("lat", "lon", "true_index"), [(59.5, -8.0, 0), (59.376, -7.876, 0), (49.626, 2.124, 1639)]
)
def test_lat_lon_to_index(
    spatial_index: geospatial.SpatialIndex, lat: float, lon: float, true_index: int
):
    index = spatial_index.lat_lon_to_index(lat, lon)
    assert index == true_index


@pytest.mark.parametrize(("lat", "lon"), [(60.0, -8.0), (59.5, -9.0), (49.5, 2.125), (49.5, 3.0)])
def test_lat_lon_to_index_raises(spatial_index: geospatial.SpatialIndex, lat: float, lon: float):
    with pytest.raises(ValueError):
        spatial_index.lat_lon_to_index(lat, lon)


@pytest.mark.parametrize(
    ("east", "west", "true_filtered"),
    [
        (-45, -90, range(360 - 90, 360 - 45)),
        (45, -45, list(range(360 - 45, 360)) + list(range(0, 45))),
    ],
)
def test_filter_longitudes_360(east: float, west: float, true_filtered: Iterable[float]):
    lons_to_filter = range(0, 360)
    bounding_box = geospatial.BoundingBox(north=60, south=50, east=east, west=west)
    filtered = bounding_box.filter_longitudes_360(lons_to_filter)
    assert filtered == list(true_filtered)


def test_round_trip_from_lat_lon_to_index_and_back(spatial_index: geospatial.SpatialIndex):
    NUM_SAMPLES: Final[int] = 512
    bbox = spatial_index.bounding_box
    rng: np.random.Generator = np.random.default_rng(seed=42)
    lats = rng.uniform(low=bbox.south, high=bbox.north, size=NUM_SAMPLES)
    lons = rng.uniform(low=bbox.west, high=bbox.east, size=NUM_SAMPLES)

    for lat, lon in zip(lats, lons):
        index = spatial_index.lat_lon_to_index(lat, lon)
        bbox_from_index = spatial_index.index_to_bounding_box(index)
        assert bbox_from_index.west <= lon <= bbox_from_index.east
        assert bbox_from_index.south <= lat <= bbox_from_index.north
