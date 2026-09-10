"""
OceanTrace - environment reader abstraction layer (Milestone 3)
===================================================================

Purpose
-------
Give the drift engine (whatever script calls OpenOil/o.run()) a single,
provider-agnostic way to obtain environmental forcing (ocean current, wind),
so that:

  * the Milestone 2 controlled-current script (test_current_displacement.py)
    is untouched and keeps using o.set_config(...) fallback values directly.
  * real environmental forcing (a local CF-compliant NetCDF subset today;
    CMEMS / ERA5 later) can be swapped in by changing one argument
    (`env_mode`), not by rewriting simulation code.
  * the simulation script never imports copernicusmarine, cdsapi, or any
    other provider-specific package directly -- only this module does.

Supported env_mode values
--------------------------
  'constant' : no readers returned; caller is expected to use
               o.set_config('environment:fallback:...') exactly as
               test_current_displacement.py already does. Included here
               only so callers can select modes uniformly.

  'netcdf'   : wraps any local (or OPeNDAP) CF-compliant NetCDF file/pattern
               using OpenDrift's own reader_netCDF_CF_generic.Reader. This
               is the path used for:
                 - the synthetic small regional/time-subset dataset built by
                   make_synthetic_environment.py (used today, see
                   test_real_environment.py)
                 - a CMEMS subset downloaded to disk with
                   copernicusmarine.subset(...) (small regional/time box)
                 - an ERA5 subset downloaded with cdsapi and converted to
                   NetCDF
               i.e. once ANY provider's data has been reduced to a small
               local CF NetCDF file, it goes through this exact same code
               path -- this is what makes the drift engine provider-agnostic.

  'cmems'    : wraps OpenDrift's built-in reader_copernicusmarine.Reader,
               which lazily opens a CMEMS dataset_id via the official
               `copernicusmarine` python client (already an OpenDrift
               dependency in this environment -- confirmed installed).
               Requires network access to marine.copernicus.eu and CMEMS
               credentials; NOT reachable from this sandbox (see README /
               milestone report), included here for completeness and for
               use in the real project environment.

API/format notes verified against the INSTALLED opendrift==1.14.12
(introspected directly, not assumed):
  - reader_netCDF_CF_generic.Reader discovers variables via the CF
    `standard_name` attribute on each NetCDF variable. Confirmed by
    direct test: a variable with standard_name='eastward_sea_water_velocity'
    (CMEMS 'uo') is exposed by the reader as both
    'eastward_sea_water_velocity' AND the internal name
    'x_sea_water_velocity' that OpenOil actually requests -- no extra
    config needed when standard_name is already correct (this is the
    case for CMEMS 'uo'/'vo').
  - If standard_name is missing or wrong (a real risk with some
    ERA5/cfgrib-converted NetCDF exports -- confirmed via cfgrib issue
    tracker), pass standard_name_mapping={'u10': 'x_wind', 'v10': 'y_wind'}
    (the FINAL internal name, not a CF alias) to the Reader constructor.
    Verified empirically: this bypasses CF-attribute lookup entirely and
    registers the variable directly.
  - reader_copernicusmarine.Reader(dataset_id, username, password) is a
    thin, first-party OpenDrift wrapper around
    reader_netCDF_CF_generic.Reader, so the mapping rules above apply to
    it too.
"""

from pathlib import Path
from typing import Optional, Sequence, Union

PathLike = Union[str, Path]


def build_constant_fallback_readers():
    """'constant' mode: no Reader objects. Caller sets
    environment:fallback:* config directly, as in test_current_displacement.py.
    Returned for API symmetry with the other build_* functions."""
    return []


def build_netcdf_readers(paths: Union[PathLike, Sequence[PathLike]],
                          standard_name_mapping: Optional[dict] = None,
                          name: Optional[str] = None):
    """'netcdf' mode: wrap one or more local/OPeNDAP CF-NetCDF files.

    :param paths: single path/URL, glob pattern, or list of paths. A single
        file can (and in our synthetic dataset does) contain both current
        (uo/vo) and wind (u10/v10) -- OpenDrift does not require separate
        files per variable group.
    :param standard_name_mapping: optional {var_name: internal_name} override
        for files that lack correct CF standard_name attributes (see module
        docstring). Use internal names directly, e.g.
        {'u10': 'x_wind', 'v10': 'y_wind'}.
    :param name: optional reader name for logging.
    """
    from opendrift.readers.reader_netCDF_CF_generic import Reader

    if isinstance(paths, (list, tuple)):
        readers = []
        for p in paths:
            readers.append(Reader(str(p), standard_name_mapping=standard_name_mapping or {},
                                   name=name))
        return readers
    else:
        return [Reader(str(paths), standard_name_mapping=standard_name_mapping or {},
                        name=name)]


def build_cmems_reader(dataset_id: str, username: Optional[str] = None,
                        password: Optional[str] = None):
    """'cmems' mode: live CMEMS dataset via OpenDrift's official wrapper.

    Requires network access to marine.copernicus.eu and valid credentials
    (via arguments, COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD env vars, or
    a .netrc entry for machine 'copernicusmarine'). NOT usable in this
    sandbox (egress is restricted to package registries only) -- provided
    for the real project environment.

    Prefer subsetting to a small regional/time box with
    `copernicusmarine.subset(...)` and loading the resulting small local
    file through build_netcdf_readers() instead, when a fixed reproducible
    test dataset is wanted rather than a live remote connection.
    """
    from opendrift.readers.reader_copernicusmarine import Reader
    return [Reader(dataset_id, username=username, password=password)]


def get_readers(env_mode: str, **kwargs):
    """Single entry point the drift engine calls. Returns a plain list of
    already-constructed OpenDrift Reader objects (possibly empty), so the
    engine code (o.add_reader(readers); o.run(...)) never needs to know
    which provider -- or whether real data was used at all -- produced them.
    """
    if env_mode == 'constant':
        return build_constant_fallback_readers()
    elif env_mode == 'netcdf':
        return build_netcdf_readers(kwargs['paths'],
                                     standard_name_mapping=kwargs.get('standard_name_mapping'),
                                     name=kwargs.get('name'))
    elif env_mode == 'cmems':
        return build_cmems_reader(kwargs['dataset_id'],
                                   username=kwargs.get('username'),
                                   password=kwargs.get('password'))
    else:
        raise ValueError(f"Unknown env_mode '{env_mode}'. "
                          f"Expected one of: 'constant', 'netcdf', 'cmems'.")
