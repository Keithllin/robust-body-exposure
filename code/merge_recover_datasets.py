#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import shutil


def list_pkls(raw_dir):
    raw_dir = Path(raw_dir).expanduser().resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw dir does not exist: {raw_dir}")
    files = sorted(raw_dir.glob('*.pkl'))
    if len(files) == 0:
        raise RuntimeError(f"no .pkl files found in raw dir: {raw_dir}")
    return files


def link_or_copy(src, dst, copy_file=False, overwrite=False):
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(f"destination already exists: {dst}")
        dst.unlink()

    if copy_file:
        try:
            shutil.copy2(str(src), str(dst))
        except (PermissionError, OSError):
            # NFS may reject chmod/metadata updates from copy2; content copy is enough.
            with open(src, 'rb') as src_handle, open(dst, 'wb') as dst_handle:
                dst_handle.write(src_handle.read())
    else:
        os.symlink(str(src), str(dst))


def materialize_files(files, output_raw_dir, prefix, limit=None, copy_file=False, overwrite=False, use_resolved_name=False):
    output_raw_dir = Path(output_raw_dir)
    selected = files if limit is None else files[:int(limit)]
    for idx, src in enumerate(selected):
        src_path = Path(src).resolve() if use_resolved_name else Path(src)
        name = src_path.name if use_resolved_name else Path(src).name
        dst = output_raw_dir / (f'{prefix}_{idx:06d}_{name}' if not use_resolved_name else name)
        link_or_copy(src_path if use_resolved_name else src, dst, copy_file=copy_file, overwrite=overwrite)
    return len(selected)


def build_parser():
    parser = argparse.ArgumentParser(description='Merge base recover raw PKLs with optional overlap/field recover raw PKLs.')
    parser.add_argument('--base-raw-dir', required=True, help='Original recover dataset raw dir, e.g. 30k raw.')
    parser.add_argument('--overlap-raw-dir', default=None, help='Overlap-guided recover dataset raw dir, e.g. 10k raw.')
    parser.add_argument('--field-raw-dir', default=None, help='Field-guided recover dataset raw dir, e.g. 5k raw.')
    parser.add_argument('--output-dataset-dir', required=True, help='Output dataset root. A raw/ directory will be created inside it.')
    parser.add_argument('--base-limit', type=int, default=None, help='Optional max number of base PKLs to include.')
    parser.add_argument('--overlap-limit', type=int, default=None, help='Optional max number of overlap PKLs to include.')
    parser.add_argument('--field-limit', type=int, default=None, help='Optional max number of field PKLs to include.')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of creating symlinks.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output raw files with matching names.')
    return parser


def main():
    args = build_parser().parse_args()
    if args.overlap_raw_dir is None and args.field_raw_dir is None:
        raise RuntimeError('Provide at least one of --overlap-raw-dir or --field-raw-dir.')

    base_files = list_pkls(args.base_raw_dir)
    overlap_files = list_pkls(args.overlap_raw_dir) if args.overlap_raw_dir else []
    field_files = list_pkls(args.field_raw_dir) if args.field_raw_dir else []

    output_dataset_dir = Path(args.output_dataset_dir).expanduser().resolve()
    output_raw_dir = output_dataset_dir / 'raw'
    output_raw_dir.mkdir(parents=True, exist_ok=True)

    base_count = materialize_files(
        base_files,
        output_raw_dir,
        'base',
        limit=args.base_limit,
        copy_file=args.copy,
        overwrite=args.overwrite,
    )
    overlap_count = 0
    if overlap_files:
        overlap_count = materialize_files(
            overlap_files,
            output_raw_dir,
            'overlap',
            limit=args.overlap_limit,
            copy_file=args.copy,
            overwrite=args.overwrite,
        )
    field_count = 0
    if field_files:
        field_count = materialize_files(
            field_files,
            output_raw_dir,
            'field',
            limit=args.field_limit,
            copy_file=args.copy,
            overwrite=args.overwrite,
        )

    summary = {
        'base_raw_dir': str(Path(args.base_raw_dir).expanduser().resolve()),
        'overlap_raw_dir': None if args.overlap_raw_dir is None else str(Path(args.overlap_raw_dir).expanduser().resolve()),
        'field_raw_dir': None if args.field_raw_dir is None else str(Path(args.field_raw_dir).expanduser().resolve()),
        'output_dataset_dir': str(output_dataset_dir),
        'output_raw_dir': str(output_raw_dir),
        'base_count': int(base_count),
        'overlap_count': int(overlap_count),
        'field_count': int(field_count),
        'total_count': int(base_count + overlap_count + field_count),
        'mode': 'copy' if args.copy else 'symlink',
    }
    with open(output_dataset_dir / 'merge_summary.json', 'w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')

    lines = [
        '# Recover Dataset Merge Summary',
        '',
        f"- base_raw_dir: `{summary['base_raw_dir']}`",
        f"- overlap_raw_dir: `{summary['overlap_raw_dir']}`",
        f"- field_raw_dir: `{summary['field_raw_dir']}`",
        f"- output_raw_dir: `{summary['output_raw_dir']}`",
        f"- base_count: {base_count}",
        f"- overlap_count: {overlap_count}",
        f"- field_count: {field_count}",
        f"- total_count: {base_count + overlap_count + field_count}",
        f"- mode: {summary['mode']}",
        '',
    ]
    with open(output_dataset_dir / 'merge_summary.md', 'w') as handle:
        handle.write('\n'.join(lines))

    print(f"Wrote merged dataset raw dir: {output_raw_dir}")
    print(f"base={base_count}, overlap={overlap_count}, field={field_count}, total={base_count + overlap_count + field_count}")


if __name__ == '__main__':
    main()
