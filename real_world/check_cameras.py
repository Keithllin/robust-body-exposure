#!/usr/bin/env python3
"""Enumerate and preview ceiling ZED camera(s).

Run:
  conda run -n robe-zed python check_cameras.py
  conda run -n robe-zed python check_cameras.py --serial <SERIAL>
"""

from __future__ import annotations

import argparse
import sys
import os.path as osp

import cv2

CODE_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from zed_util import grab_rgb_bgr, list_zed_serials, open_ceiling_zed, warm_up  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Check / preview ZED cameras")
    parser.add_argument("--serial", type=int, default=0, help="ZED serial; 0 = first")
    parser.add_argument("--list-only", action="store_true", help="Only list serials")
    args = parser.parse_args()

    serials = list_zed_serials()
    print(f"Found {len(serials)} ZED device(s): {serials}")
    if args.list_only:
        return
    if not serials:
        raise SystemExit("No ZED cameras detected. Check USB / SDK.")

    serial = args.serial or serials[0]
    print(f"Opening ZED serial={serial} (q to quit preview)...")
    cam = open_ceiling_zed(serial=serial)
    try:
        warm_up(cam, 15)
        while True:
            bgr = grab_rgb_bgr(cam)
            if bgr is None:
                continue
            cv2.putText(
                bgr,
                f"ZED SN {serial}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("ZED check_cameras", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()
    print("OK")


if __name__ == "__main__":
    main()
