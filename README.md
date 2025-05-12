# clean_uk_pv
Cleaning the solar power dataset [uk_pv](https://huggingface.co/datasets/openclimatefix/uk_pv).

Todo:
- [ ] Create script to download ARCO-ERA5 for the UK, for the variables required for pvlib, for 2010 to 2025.
- [ ] Remove timesteps in `bad_data.csv`, and -ve values, and too-high values, and values either side of each "insane" value.
- [ ] "Predict" PV power for each PV system using PVLib, ERA5, and the metadata.
- [ ] For each PV system, and for each day: Remove if PV power is non-zero at night, or if PV power is zero during day, or if PV power doesn't correlate with PVLib's prediction.
- [ ] For each PV system, check the metadata by checking agreement with PVLib's prediction (using ERA5).

