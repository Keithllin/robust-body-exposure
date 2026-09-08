import argparse
import os.path as osp
import subprocess
import sys

CODE_DIR = osp.dirname(osp.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from conda_python import env_for_python, robe_python, zed_python  # noqa: E402
from pickle_compat import load_pickle  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Save spectral data")
    parser.add_argument("--subject-dir", type=str, default="TEST")
    parser.add_argument("--pose-dir", type=str, default="TEST")
    parser.add_argument("--manikin", type=int, default=0)
    parser.add_argument(
        "--serial",
        type=int,
        default=0,
        help="Ceiling ZED serial; 0 lets the ZED SDK choose a camera",
    )
    parser.add_argument(
        "--skip-aruco",
        action="store_true",
        default=True,
        help="Skip ArUco origin (default True until board is ready)",
    )
    parser.add_argument(
        "--require-aruco",
        action="store_true",
        help="Require ArUco sim_origin_data.pkl (disables --skip-aruco)",
    )
    parser.add_argument(
        "--remeasure-body",
        action="store_true",
        help="Ignore existing body_info.pkl and re-run the 9 diameter GUI "
        "(UPPERCHEST then WAIST are two spheres). Does not re-run MediaPipe.",
    )
    parser.add_argument(
        "--recapture-pose",
        action="store_true",
        help="Re-capture uncovered RGB / ArUco and re-run MediaPipe even if "
        "uncovered_rgb.png and human_pose.pkl already exist.",
    )
    parser.add_argument(
        "--reuse-subject-body",
        action="store_true",
        help="If this trial and this exp's pose pack have no body_info.pkl, "
        "copy subject_dir/body_info.pkl. Not used unless this flag is set.",
    )
    args = parser.parse_args()

    zed_py = zed_python()
    robe_py = robe_python()
    skip_aruco = not args.require_aruco

    def _call(cmd):
        subprocess.check_call(cmd, env=env_for_python(cmd[0]))

    def valid_body_info(path):
        required_parts = (
            "head",
            "upperchest",
            "waist",
            "upperarm",
            "forearm",
            "hand",
            "thigh",
            "shin",
            "foot",
        )
        torso_min_radius = 0.06  # 12 cm diameter; two full upper-body spheres
        try:
            if args.remeasure_body:
                return False
            data = load_pickle(path)
            for name in required_parts:
                if name not in data or len(data[name]) < 2:
                    return False
                radius = float(data[name][1])
                need = torso_min_radius if name in ("upperchest", "waist") else 0.005
                if radius < need:
                    return False
            return True
        except (OSError, EOFError, KeyError, TypeError, ValueError):
            return False

    uncover = osp.join(CODE_DIR, "capture_pose", "get_uncovered_img_and_origin.py")
    body_info = osp.join(CODE_DIR, "capture_pose", "get_body_info_from_img.py")
    pose_gui = osp.join(CODE_DIR, "capture_pose", "mediapipe_pose_detect_gui.py")

    def uncover_command(save_dir):
        command = [zed_py, uncover, "--save-dir", save_dir]
        if args.serial:
            command.extend(["--serial", str(args.serial)])
        if skip_aruco:
            command.append("--skip-aruco")
        return command

    if args.manikin:
        if not osp.exists(osp.join(args.subject_dir, "uncovered_rgb.png")):
            cmd = uncover_command(args.subject_dir)
            print(f"Running: {' '.join(cmd)}")
            _call(cmd)
        body_path = osp.join(args.subject_dir, "body_info.pkl")
        if not valid_body_info(body_path):
            if skip_aruco and not osp.exists(osp.join(args.subject_dir, "sim_origin_data.pkl")):
                print(
                    "WARN: body_info GUI (9 diameters, including WAIST as "
                    "upper-body sphere 2/2) needs sim_origin_data.pkl; "
                    "skipping until ArUco board is ready (--require-aruco). "
                    "Without WAIST the human model is one chest disk and one point."
                )
            else:
                _call(
                    [
                        robe_py,
                        body_info,
                        "--subject-dir",
                        args.subject_dir,
                        "--pose-dir",
                        args.subject_dir,
                    ]
                )
                if not osp.exists(body_path):
                    raise RuntimeError(
                        "body_info.pkl was not created. Complete all nine "
                        "manual body-diameter measurements (UPPERCHEST then "
                        "WAIST are two separate spheres) before closing the GUI."
                    )
        else:
            print(f"Reusing existing body diameters: {body_path}")
            print("Pass --remeasure-body to draw the 9 lines again.")
        if not osp.exists(osp.join(args.subject_dir, "human_pose.pkl")):
            if skip_aruco and not osp.exists(osp.join(args.subject_dir, "sim_origin_data.pkl")):
                print(
                    "WARN: MediaPipe pose needs sim_origin_data.pkl; "
                    "skipping until ArUco board is ready."
                )
            else:
                _call(
                    [
                        zed_py,
                        pose_gui,
                        "--subject-dir",
                        args.subject_dir,
                        "--pose-dir",
                        args.subject_dir,
                    ]
                )
    else:
        from pathlib import Path

        from trial_layout import ensure_trial_body_info

        rgb_path = osp.join(args.pose_dir, "uncovered_rgb.png")
        origin_path = osp.join(args.pose_dir, "sim_origin_data.pkl")
        need_uncover = (
            args.recapture_pose
            or not osp.exists(rgb_path)
            or (not skip_aruco and not osp.exists(origin_path))
        )
        if need_uncover:
            cmd = uncover_command(args.pose_dir)
            print(f"Running: {' '.join(cmd)}")
            _call(cmd)
        if not skip_aruco and not osp.exists(origin_path):
            raise RuntimeError(
                f"Missing {origin_path}. Uncovered ArUco capture must write "
                "sim_origin_data.pkl before body diameters or MediaPipe."
            )

        ensure_trial_body_info(
            Path(args.pose_dir),
            Path(args.subject_dir),
            allow_subject=bool(args.reuse_subject_body),
        )
        body_path = osp.join(args.pose_dir, "body_info.pkl")
        if not valid_body_info(body_path):
            if skip_aruco and not osp.exists(origin_path):
                print(
                    "WARN: body_info GUI (9 diameters, including WAIST as "
                    "upper-body sphere 2/2) needs sim_origin_data.pkl; "
                    "skipping until ArUco board is ready (--require-aruco). "
                    "Without WAIST the human model is one chest disk and one point."
                )
            else:
                _call(
                    [
                        robe_py,
                        body_info,
                        "--subject-dir",
                        args.subject_dir,
                        "--pose-dir",
                        args.pose_dir,
                    ]
                )
                if not osp.exists(body_path):
                    raise RuntimeError(
                        "body_info.pkl was not created. Complete all nine "
                        "manual body-diameter measurements (UPPERCHEST then "
                        "WAIST are two separate spheres) before closing the GUI."
                    )
        else:
            print(f"Reusing existing body diameters: {body_path}")
            print("Pass --remeasure-body to draw the 9 lines again.")
        human_pose_path = osp.join(args.pose_dir, "human_pose.pkl")
        if osp.exists(human_pose_path) and not args.recapture_pose:
            print(
                f"Reusing existing human pose: {human_pose_path}; "
                "pass --recapture-pose to re-run MediaPipe."
            )
        elif skip_aruco and not osp.exists(osp.join(args.pose_dir, "sim_origin_data.pkl")):
            print(
                "WARN: MediaPipe pose needs sim_origin_data.pkl; "
                "skipping until ArUco board is ready."
            )
        else:
            _call(
                [
                    zed_py,
                    pose_gui,
                    "--subject-dir",
                    args.subject_dir,
                    "--pose-dir",
                    args.pose_dir,
                ]
            )


if __name__ == "__main__":
    main()
