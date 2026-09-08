"""Session resolve-once, freeze-contract, and dependency-invalidation checks."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from session_paths import (
    DEFAULT_LAYOUT,
    STATUS_FROZEN,
    STATUS_MISSING,
    STATUS_NEEDS_ALIGNMENT,
    STATUS_RAW_READY,
    SessionPaths,
    SessionRegistrationError,
    assert_registration_frozen,
    default_session_contract,
    evaluate_session,
    freeze_registration,
    freeze_zed,
    mark_raw_arrived,
    receive_stretch_raw,
    resolve_session_dir_once,
    save_session_json,
    sha256_file,
    write_pcd_config,
)


def _raw_payload(**extra):
    payload = {
        "ok": True,
        "T_odom_layout": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    }
    payload.update(extra)
    return payload


class SessionPathsTest(unittest.TestCase):
    def _require_local_layout(self) -> None:
        if not DEFAULT_LAYOUT.is_file():
            self.skipTest(
                "local marker_layout.json is required for registration hash tests"
            )

    def test_resolve_once_ignores_symlink_change(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "20260828_a"
            b = root / "20260828_b"
            a.mkdir()
            b.mkdir()
            link = root / "current"
            link.symlink_to(a.name)
            first = resolve_session_dir_once(link, force=True)
            self.assertEqual(first, a.resolve())
            link.unlink()
            link.symlink_to(b.name)
            second = resolve_session_dir_once(link, force=False)
            self.assertEqual(second, a.resolve())

    def test_new_contract_statuses_are_missing(self) -> None:
        contract = default_session_contract("sess")
        self.assertEqual(contract["zed_status"], STATUS_MISSING)
        self.assertEqual(contract["stretch_origin_status"], STATUS_MISSING)
        self.assertEqual(contract["registration_status"], STATUS_MISSING)

    def test_raw_arrival_invalidates_old_corrected(self) -> None:
        self._require_local_layout()
        with TemporaryDirectory() as tmp:
            session = Path(tmp) / "sess"
            paths = SessionPaths(session)
            paths.stretch_dir.mkdir(parents=True)
            paths.zed_dir.mkdir(parents=True)
            paths.zed_extrinsics.write_text('{"cameras":{}}\n')
            write_pcd_config(session)
            save_session_json(session, default_session_contract("sess"))
            freeze_zed(session)
            raw = _raw_payload()
            paths.origin_raw.write_text(json.dumps(raw) + "\n")
            paths.origin_corrected.write_text(json.dumps(raw) + "\n")
            registration = {
                "reference": "aruco_136",
                "accepted": True,
                "before_xy_error_m": 0.052,
                "after_xy_error_m": 0.0,
                "stretch_origin_raw_sha256": sha256_file(paths.origin_raw),
                "zed_extrinsics_sha256": sha256_file(paths.zed_extrinsics),
                "marker_layout_sha256": sha256_file(DEFAULT_LAYOUT),
            }
            freeze_registration(
                session,
                xy_registration_m=[-0.05, 0.0],
                residual_xy_m=0.0,
                registration=registration,
            )
            assert_registration_frozen(session)
            raw2 = _raw_payload(layout_origin_odom_m=[1.0, 2.0, 3.0])
            tmp_json = paths.stretch_dir / "stretch_origin.json.tmp"
            tmp_json.write_text(json.dumps(raw2) + "\n")
            receive_stretch_raw(paths.stretch_dir)
            contract = json.loads(paths.session_json.read_text())
            self.assertEqual(contract["registration_status"], STATUS_NEEDS_ALIGNMENT)
            self.assertEqual(contract["stretch_origin_status"], STATUS_RAW_READY)
            with self.assertRaises(SessionRegistrationError):
                assert_registration_frozen(session)
            self.assertTrue(paths.origin_corrected.is_file())

    def test_new_zed_freeze_invalidates_registration(self) -> None:
        self._require_local_layout()
        with TemporaryDirectory() as tmp:
            session = Path(tmp) / "sess"
            paths = SessionPaths(session)
            paths.stretch_dir.mkdir(parents=True)
            paths.zed_dir.mkdir(parents=True)
            paths.zed_extrinsics.write_text('{"cameras":{"ceiling":{}}}\n')
            write_pcd_config(session)
            save_session_json(session, default_session_contract("sess"))
            paths.origin_raw.write_text(json.dumps(_raw_payload()) + "\n")
            mark_raw_arrived(session)
            paths.origin_corrected.write_text(json.dumps(_raw_payload()) + "\n")
            freeze_registration(
                session,
                xy_registration_m=[0.0, 0.0],
                residual_xy_m=0.0,
                registration={
                    "reference": "aruco_136",
                    "accepted": True,
                    "stretch_origin_raw_sha256": sha256_file(paths.origin_raw),
                    "zed_extrinsics_sha256": sha256_file(paths.zed_extrinsics),
                    "marker_layout_sha256": sha256_file(DEFAULT_LAYOUT),
                    "after_xy_error_m": 0.0,
                },
            )
            assert_registration_frozen(session)
            paths.zed_extrinsics.write_text('{"cameras":{"ceiling":{"serial":1}}}\n')
            freeze_zed(session)
            contract = json.loads(paths.session_json.read_text())
            self.assertEqual(contract["zed_status"], STATUS_FROZEN)
            self.assertEqual(contract["registration_status"], STATUS_NEEDS_ALIGNMENT)
            with self.assertRaises(SessionRegistrationError):
                assert_registration_frozen(session)

    def test_registration_hash_drift_fails_ready(self) -> None:
        self._require_local_layout()
        with TemporaryDirectory() as tmp:
            session = Path(tmp) / "sess"
            paths = SessionPaths(session)
            paths.stretch_dir.mkdir(parents=True)
            paths.zed_dir.mkdir(parents=True)
            paths.zed_extrinsics.write_text('{"cameras":{}}\n')
            write_pcd_config(session)
            save_session_json(session, default_session_contract("sess"))
            freeze_zed(session)
            paths.origin_raw.write_text(json.dumps(_raw_payload()) + "\n")
            mark_raw_arrived(session)
            paths.origin_corrected.write_text(json.dumps(_raw_payload()) + "\n")
            freeze_registration(
                session,
                xy_registration_m=[0.0, 0.0],
                residual_xy_m=0.0,
                registration={
                    "reference": "aruco_136",
                    "accepted": True,
                    "stretch_origin_raw_sha256": sha256_file(paths.origin_raw),
                    "zed_extrinsics_sha256": sha256_file(paths.zed_extrinsics),
                    "marker_layout_sha256": sha256_file(DEFAULT_LAYOUT),
                    "after_xy_error_m": 0.0,
                },
            )
            self.assertTrue(evaluate_session(session).ready)
            paths.zed_extrinsics.write_text('{"cameras":{"stale":true}}\n')
            report = evaluate_session(session)
            self.assertFalse(report.ready)
            self.assertFalse(report.registration_ok)

    def test_pcd_config_uses_relative_filter_names(self) -> None:
        with TemporaryDirectory() as tmp:
            session = Path(tmp) / "sess"
            SessionPaths(session).zed_dir.mkdir(parents=True)
            path = write_pcd_config(session)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["filter_json"]["ceiling"], "zed_blanket_filter.json")
            self.assertFalse(Path(payload["filter_json"]["ceiling"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
