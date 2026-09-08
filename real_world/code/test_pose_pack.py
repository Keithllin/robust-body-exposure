"""Same pose_num TLs copy from the current session pose pack."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from trial_layout import (  # noqa: E402
    POSE_PACK_FILES,
    apply_pose_pack,
    ensure_trial_body_info,
    export_pose_pack_to_session,
    pose_pack_complete,
    resolve_body_info,
    resolve_pose_pack_source,
    session_pose_dir,
)


def _touch_pack(
    directory: Path, *, skip: str | None = None, body: bool = False, viz: bool = False
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in POSE_PACK_FILES:
        if name == skip:
            continue
        (directory / name).write_text(name)
    if body:
        (directory / "body_info.pkl").write_text("body")
    if viz:
        (directory / "all_body_points_over_rgb.png").write_text("viz")


class PosePackTest(unittest.TestCase):
    def test_legacy_sibling_copy_without_session(self) -> None:
        with TemporaryDirectory() as tmp:
            subject = Path(tmp) / "subject_smoke"
            old = subject / "pose_1_TL2_111"
            new = subject / "pose_1_TL2_222"
            _touch_pack(old)
            source, copied = apply_pose_pack(new, subject, "1")
            self.assertEqual(source, old)
            self.assertTrue(pose_pack_complete(new))
            self.assertIn("human_pose.pkl", copied)
            self.assertIn("sim_origin_data.pkl", copied)
            self.assertFalse((subject / "pose_1").exists())

    def test_session_pack_copies_body_and_viz(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = root / "subject_smoke"
            session = root / "sessions" / "exp000"
            pack = session_pose_dir(session, "1")
            new = subject / "pose_1_TL2_222"
            other = subject / "pose_1_TL2_111"
            _touch_pack(pack, body=True, viz=True)
            _touch_pack(other, body=True)
            (other / "human_pose.pkl").write_text("OTHER")
            source, copied = apply_pose_pack(
                new, subject, "1", session_dir=session
            )
            self.assertEqual(source, pack)
            self.assertTrue(pose_pack_complete(new, require_body=True))
            self.assertIn("body_info.pkl", copied)
            self.assertIn("all_body_points_over_rgb.png", copied)
            self.assertEqual((new / "human_pose.pkl").read_text(), "human_pose.pkl")
            self.assertEqual(
                resolve_body_info(new, subject, session_dir=session),
                new / "body_info.pkl",
            )

    def test_other_exp_sibling_is_not_used_when_session_empty(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = root / "subject_smoke"
            session = root / "sessions" / "exp000"
            old = subject / "pose_1_TL2_111"
            new = subject / "pose_1_TL2_222"
            _touch_pack(old, body=True, viz=True)
            source, copied = apply_pose_pack(
                new, subject, "1", session_dir=session
            )
            self.assertIsNone(source)
            self.assertEqual(copied, [])
            self.assertFalse(pose_pack_complete(new))

    def test_export_then_reuse(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = root / "subject_smoke"
            session = root / "sessions" / "exp000"
            first = subject / "pose_1_TL2_111"
            second = subject / "pose_1_TL2_222"
            _touch_pack(first, body=True, viz=True)
            exported = export_pose_pack_to_session(first, session, "1")
            self.assertIn("human_pose.pkl", exported)
            self.assertIn("body_info.pkl", exported)
            self.assertIn("all_body_points_over_rgb.png", exported)
            source, copied = apply_pose_pack(
                second, subject, "1", session_dir=session
            )
            self.assertEqual(source, session_pose_dir(session, "1"))
            self.assertTrue(pose_pack_complete(second, require_body=True))
            self.assertTrue((second / "all_body_points_over_rgb.png").is_file())

    def test_no_sibling_copy_when_disabled(self) -> None:
        with TemporaryDirectory() as tmp:
            subject = Path(tmp) / "subject_smoke"
            old = subject / "pose_1_TL2_111"
            new = subject / "pose_1_TL2_222"
            _touch_pack(old)
            source, copied = apply_pose_pack(
                new, subject, "1", copy_sibling=False
            )
            self.assertIsNone(source)
            self.assertEqual(copied, [])
            self.assertFalse(pose_pack_complete(new))

    def test_incomplete_sibling_is_not_a_source(self) -> None:
        with TemporaryDirectory() as tmp:
            subject = Path(tmp) / "subject_smoke"
            old = subject / "pose_1_TL2_111"
            _touch_pack(old, skip="sim_origin_data.pkl")
            self.assertIsNone(resolve_pose_pack_source(subject, "1"))
            source, copied = apply_pose_pack(subject / "pose_1_TL2_222", subject, "1")
            self.assertIsNone(source)
            self.assertEqual(copied, [])

    def test_subject_body_is_not_a_silent_fallback(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = root / "subject_smoke"
            session = root / "sessions" / "exp000"
            trial = subject / "pose_1_TL2_222"
            trial.mkdir(parents=True)
            session_pose_dir(session, "1").mkdir(parents=True)
            (subject / "body_info.pkl").write_text("OLD_EXP")
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_body_info(trial, subject, session_dir=session)
            self.assertIn("--reuse-subject-body", str(ctx.exception))
            self.assertIsNone(
                ensure_trial_body_info(trial, subject, session_dir=session)
            )
            self.assertFalse((trial / "body_info.pkl").is_file())

    def test_reuse_subject_body_is_explicit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = root / "subject_smoke"
            session = root / "sessions" / "exp000"
            trial = subject / "pose_1_TL2_222"
            trial.mkdir(parents=True)
            session_pose_dir(session, "1").mkdir(parents=True)
            (subject / "body_info.pkl").write_text("OLD_EXP")
            path = resolve_body_info(
                trial, subject, session_dir=session, allow_subject=True
            )
            self.assertEqual(path, subject / "body_info.pkl")
            copied = ensure_trial_body_info(
                trial, subject, session_dir=session, allow_subject=True
            )
            self.assertEqual(copied, trial / "body_info.pkl")
            self.assertEqual((trial / "body_info.pkl").read_text(), "OLD_EXP")

    def test_session_pack_body_is_used_without_subject_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = root / "subject_smoke"
            session = root / "sessions" / "exp000"
            trial = subject / "pose_1_TL2_222"
            pack = session_pose_dir(session, "1")
            trial.mkdir(parents=True)
            pack.mkdir(parents=True)
            (pack / "body_info.pkl").write_text("THIS_EXP")
            (subject / "body_info.pkl").write_text("OLD_EXP")
            path = resolve_body_info(trial, subject, session_dir=session)
            self.assertEqual(path, pack / "body_info.pkl")
            copied = ensure_trial_body_info(trial, subject, session_dir=session)
            self.assertEqual(copied, trial / "body_info.pkl")
            self.assertEqual((trial / "body_info.pkl").read_text(), "THIS_EXP")


if __name__ == "__main__":
    unittest.main()
