# ZED calibration artifacts

Robot↔planning origin is **session registration**, not this ZED calib.
See `real_world/SESSION_REGISTRATION.md`.

Copy `marker_layout.example.json` to `marker_layout.json` locally; the local
file is the single source of truth for the bed frame. Fill
`center_m`, `u_axis`, and `v_axis` for IDs `0, 1, 2, 3, 10, 11, 12, 13`
before running any PnP calibration.

- `center_m` is the marker center in meters in the bed frame.
- `u_axis` points from the printed top-left corner to top-right.
- `v_axis` points from the printed top-left corner to bottom-left.
- Every tag is `0.07` m and uses `DICT_5X5_100`.
- ZED XYZ is captured in `COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP`; the
  calibration code explicitly converts the OpenCV image-frame PnP pose before
  saving `T_bed_camera`.

Use the same physical layout for all three cameras. The calibration command
writes one role into `zed_extrinsics.json`:

```bash
conda run -n robe-zed python calibration/calibrate_zed_markers.py \
  --role ceiling --layout calibration/marker_layout.json --check-layout
conda run -n robe-zed python calibration/calibrate_zed_markers.py \
  --role ceiling --layout calibration/marker_layout.json --serial <SERIAL> --sim-origin-dir <POSE_DIR> \
  --lock-exposure-gain
conda run -n robe-zed python calibration/calibrate_zed_markers.py \
  --role side_left --layout calibration/marker_layout.json --serial <SERIAL> --lock-exposure-gain
conda run -n robe-zed python calibration/calibrate_zed_markers.py \
  --role side_right --layout calibration/marker_layout.json --serial <SERIAL> --lock-exposure-gain
```

Side cameras require their own tuned filter files:

```text
zed_blanket_filter_side_left.json
zed_blanket_filter_side_right.json
```

They must not reuse the ceiling depth window unless the measured camera
geometry really gives the same depth range.

Keep side-camera filter settings in local, untracked configuration files.
Do not copy measured filters or camera serials into the repository.

## Exposure/gain lock

`--lock-exposure-gain` first lets auto exposure settle during `--warmup`, then
disables ZED AEC/AGC and freezes the selected values. The chosen values are
saved in `zed_extrinsics.json` with each camera role and are reused by the
point-cloud capture script. To force fixed values instead, pass for example:

```bash
conda run -n robe-zed python calibration/calibrate_zed_markers.py \
  --role ceiling --serial <SERIAL> --exposure 35 --gain 25
```

Both values use ZED's 0--100 percentage scale. The filter tuner and point-cloud
capture script accept the same `--lock-exposure-gain`, `--exposure`, and `--gain`
options.
