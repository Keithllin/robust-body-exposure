import argparse
import os.path as osp
import subprocess
import sys

CODE_DIR = osp.dirname(osp.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from conda_python import env_for_python, zed_python  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--subject-dir", type=str, default="TEST")
    parser.add_argument("--pose-dir", type=str, default="TEST")
    parser.add_argument("--manikin", type=str, default="0")
    parser.add_argument(
        "--roles",
        type=str,
        default="ceiling",
        help="Comma-separated ZED roles: ceiling,side_left,side_right",
    )
    parser.add_argument("--extrinsics", type=str, default=None)
    parser.add_argument("--require-calibration", action="store_true")
    parser.add_argument("--allow-image-frame", action="store_true")
    parser.add_argument("--forbid-calibration-fallback", action="store_true")
    parser.add_argument("--bed-x-min", type=float, default=-0.55)
    parser.add_argument("--bed-x-max", type=float, default=0.55)
    parser.add_argument("--bed-y-min", type=float, default=-1.10)
    parser.add_argument("--bed-y-max", type=float, default=1.10)
    parser.add_argument(
        "--merge-mode",
        choices=("auto", "union", "top_supported", "ceiling_primary"),
        default="auto",
    )
    parser.add_argument(
        "--support-radius-3d",
        "--support-radius-xy",
        dest="support_radius",
        type=float,
        default=0.03,
        help="XYZ support radius in meters for pairwise merging",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=None,
        help="PCD lock exposure 0-100; default is capture_and_merge_pcds default",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=None,
        help="PCD lock gain 0-100; default is capture_and_merge_pcds default",
    )
    parser.add_argument("--filter-json", type=str, default=None)
    parser.add_argument("--filter-json-side-left", type=str, default=None)
    parser.add_argument("--filter-json-side-right", type=str, default=None)
    parser.add_argument(
        "--mask-backend",
        choices=("sam2", "hsv"),
        default="sam2",
        help="sam2: HSV box then SAM2 on all roles (default). hsv: color filter only.",
    )
    parser.add_argument(
        "--detect-top-markers",
        action="store_true",
        help="Re-detect bed-corner markers 0-3. Default: reuse canonical frame.",
    )
    args = parser.parse_args()

    script = osp.join(CODE_DIR, "capture_blanket", "capture_and_merge_pcds.py")
    py = zed_python()
    print(f"Using interpreter: {py}")
    cmd = [
        py,
        script,
        "--subject-dir",
        args.subject_dir,
        "--pose-dir",
        args.pose_dir,
        "--manikin",
        str(args.manikin),
        "--roles",
        args.roles,
        "--bed-x-min",
        str(args.bed_x_min),
        "--bed-x-max",
        str(args.bed_x_max),
        "--bed-y-min",
        str(args.bed_y_min),
        "--bed-y-max",
        str(args.bed_y_max),
        "--merge-mode",
        args.merge_mode,
        "--support-radius-3d",
        str(args.support_radius),
    ]
    if args.extrinsics:
        cmd.extend(["--extrinsics", args.extrinsics])
    if args.require_calibration:
        cmd.append("--require-calibration")
    if args.allow_image_frame:
        cmd.append("--allow-image-frame")
    if args.forbid_calibration_fallback:
        cmd.append("--forbid-calibration-fallback")
    if args.exposure is not None:
        cmd.extend(["--exposure", str(int(args.exposure))])
    if args.gain is not None:
        cmd.extend(["--gain", str(int(args.gain))])
    if args.filter_json:
        cmd.extend(["--filter-json", args.filter_json])
    if args.filter_json_side_left:
        cmd.extend(["--filter-json-side-left", args.filter_json_side_left])
    if args.filter_json_side_right:
        cmd.extend(["--filter-json-side-right", args.filter_json_side_right])
    cmd.extend(["--mask-backend", args.mask_backend])
    if args.detect_top_markers:
        cmd.append("--detect-top-markers")
    subprocess.check_call(cmd, env=env_for_python(py))


if __name__ == "__main__":
    main()
