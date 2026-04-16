# Make a conda envoriment:
```
name: bachelor
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pip
  - ipykernel
  - ipywidgets
  - nbformat>=4.2.0
  - numpy=1.26.4
  - scipy
  - pandas
  - matplotlib
  - xarray
  - netcdf4
  - h5netcdf
  - h5py
  - xesmf
  - plotly
  - pip:
    - -e . # Install directory as a package; `from Tracer import ...`
```

# To create the environment, run:
```conda env create -f env.yml```
# To activate the environment, run:
```conda activate bachelor```
# To update the environment, run:
```conda env update -f env.yml --prune```
