import pytest
from clean_uk_pv import geospatial


@pytest.fixture
def spatial_index() -> geospatial.SpatialIndex:
    """
    Get a spatial index for Great Britain with a resolution of 0.25 degrees.
    """
    return geospatial.SpatialIndex(geospatial.GBBoundingBox(), 0.25)


def test_init_gb_bouding_box():
    gb_bounding_box = geospatial.GBBoundingBox()
    assert gb_bounding_box.north == 59.625
    assert gb_bounding_box.south == 49.625
    assert gb_bounding_box.east == 2.125
    assert gb_bounding_box.west == -8.125
    assert gb_bounding_box.min_lat == 49.625
    assert gb_bounding_box.max_lat == 59.625
    assert gb_bounding_box.min_lon == -8.125
    assert gb_bounding_box.max_lon == 2.125
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
