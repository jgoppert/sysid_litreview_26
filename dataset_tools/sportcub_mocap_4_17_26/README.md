# Sport Cub MoCap 4/17/26 Dataset

This is the provisional dataset package for the Sport Cub S 2 real-flight
motion-capture data contributed on the `real_data` branch.

The large raw data is not committed to git. The current source is the temporary
Purdue SharePoint folder recorded in `dataset.json`. Downloaded raw files should
live outside git under:

```text
work/data/sportcub_mocap_4_17_26/raw/
```

The compact benchmark artifact produced from those raw files should live under:

```text
data/sportcub_mocap_4_17_26_train.npz
data/sportcub_mocap_4_17_26_validation.npz
```

Useful commands from the repository root:

```bash
./results.py process-dataset sportcub_mocap_4_17_26
./results.py canonicalize-dataset sportcub_mocap_4_17_26
./results.py check-data sportcub_mocap_4_17_26
./results.py sportcub-real
./results.py web-data
```

`fetch-dataset` requires a direct archive URL. The provisional Purdue
SharePoint source is a folder link, so download it manually to
`work/data/sportcub_mocap_4_17_26/raw/` unless a direct `.zip` URL is provided.

`sportcub-real` exports the latest Sport Cub grey-box OEM result CSV into
`results/sportcub_mocap_4_17_26_method_comparison.csv`, which is then
included in the static website data bundle.
