"""CSV/JSON indexing and combination pipeline for disjoint SBM results.

Migrated from tmp/hedonic/scripts/data_reader.py.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from hedonic.experiments.config import DISJOINT_ARTIFACTS_DIR, expand_path

WORKERS = 16


def extract_sorting_keys(filename: str):
    """Extract sorting keys from a result path or filename."""
    match = re.search(r"network_(\d+)\.", filename)
    network_index = int(match.group(1)) if match else float("inf")
    if network_index == float("inf"):
        match = re.findall(r"Network \(0*(\d+)\)", filename)
        network_index = int(match[0]) if match else float("inf")
    if network_index == float("inf"):
        raise ValueError(f"Could not find network index in filename: `{filename}`")

    difficulty_match = re.search(r"Difficulty = (\d+\.\d+)", filename)
    difficulty = float(difficulty_match.group(1)) if difficulty_match else float("inf")

    p_in_match = re.search(r"P_in = (\d+\.\d+)", filename)
    p_in = float(p_in_match.group(1)) if p_in_match else float("inf")

    n_communities_match = re.search(r"(\d+)C_", filename)
    n_communities = (
        int(n_communities_match.group(1)) if n_communities_match else float("inf")
    )

    noise_match = re.search(r"Noise = (\d+\.\d+)", filename)
    noise = float(noise_match.group(1)) if noise_match else float("inf")

    return (network_index, n_communities, p_in, difficulty, -noise)


def sort_files(file_list: list[str]) -> list[str]:
    """Sort file paths by network seed and experiment settings."""
    return sorted(file_list, key=extract_sorting_keys)


def collect_paths(leaf_folder: str, extension: str = ".json") -> list[str]:
    """Collect files with the given extension from a leaf folder (no subdirs)."""
    file_paths = []
    try:
        for file in os.listdir(leaf_folder):
            if file.endswith(extension):
                full_path = os.path.join(leaf_folder, file)
                if os.path.isfile(full_path):
                    file_paths.append(full_path)
    except Exception as e:
        print(f"Error processing folder {leaf_folder}: {e}")
    return file_paths


def get_leaf_subdirs_from_subroot(subroot: str) -> list[str]:
    """Walk a subtree and return all leaf directories."""
    leaf_dirs = []
    for dirpath, dirs, _ in os.walk(subroot):
        if not dirs:
            leaf_dirs.append(dirpath)
    return leaf_dirs


def get_subroots(root: str, max_depth: int = 1) -> list[str]:
    """Return subdirectories from root at a given relative depth."""
    subroots = []
    for dirpath, dirs, _ in tqdm(
        os.walk(root), desc="Walking directory tree for subroots", leave=False
    ):
        rel_depth = dirpath[len(root) :].count(os.sep)
        if rel_depth == max_depth:
            subroots.append(dirpath)
            dirs[:] = []
    return subroots


def get_leaf_subdirs_parallel(root: str, subroot_depth: int = 4) -> list[str]:
    """Collect leaf directories in parallel from subroots at a fixed depth."""
    subroots = get_subroots(root, max_depth=subroot_depth)
    if not subroots:
        subroots = [root]
    try:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            results = list(
                tqdm(
                    executor.map(get_leaf_subdirs_from_subroot, subroots),
                    total=len(subroots),
                    desc="Get leaf subdirs from subroot paths",
                )
            )
    except (NotImplementedError, OSError, PermissionError):
        # Some constrained runners (including macOS sandboxes) expose no
        # POSIX semaphore namespace.  Directory discovery is I/O-bound and
        # deterministic, so a sequential fallback preserves the CLI contract.
        results = [get_leaf_subdirs_from_subroot(path) for path in subroots]
    return [leaf for sublist in results for leaf in sublist]


def collect_paths_helper(args: tuple[str, str]) -> list[str]:
    leaf_folder, extension = args
    return collect_paths(leaf_folder, extension)


def get_paths_sorted(folder_path: str, extension: str = ".json") -> list[str]:
    """List files with extension under leaf dirs, sorted by experiment keys."""
    leaf_dirs = get_leaf_subdirs_parallel(folder_path)
    args_list = [(lf, extension) for lf in leaf_dirs]
    try:
        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            results = list(
                tqdm(
                    executor.map(collect_paths_helper, args_list),
                    total=len(leaf_dirs),
                    desc=f"Collecting {extension} paths",
                )
            )
    except (NotImplementedError, OSError, PermissionError):
        results = [collect_paths_helper(args) for args in args_list]
    file_paths = [file for sublist in results for file in sublist]
    print(f"Found {len(file_paths)} {extension} files to be sorted.")
    return sort_files(file_paths)


def split_json_paths(json_paths: list[str]) -> list[list[str]]:
    """Split paths into groups keyed by Noise value in the path."""
    temp_list, final_list = [], []
    last_noise = None
    for fp in tqdm(json_paths, desc="Splitting JSON paths"):
        matches = re.findall(r"Noise = (\d+\.\d+)", fp)
        if not matches:
            continue
        noise = float(matches[0])
        if last_noise is None:
            last_noise = noise
        if noise != last_noise:
            final_list.append(temp_list)
            temp_list = []
            last_noise = noise
        temp_list.append(fp)
    if temp_list:
        final_list.append(temp_list)
    return final_list


def dump_json_paths(json_paths: list[str], output_file: str, verbose: bool = False) -> None:
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for line in json_paths:
            f.write(line + "\n")
    if verbose:
        print(f"Saved list of file paths to: `{output_file}`")


def dump_json_paths_wrapper(args: tuple[list[str], str, bool]) -> str:
    json_list, output_file, verbose = args
    dump_json_paths(json_list, output_file, verbose)
    return output_file


def _sibling_output_path(fp: str, results_token: str, output_folder: str) -> str:
    """Map a results path to a sibling folder (resultados → json_paths/csv_results).

    Supports both POSIX ``/resultados/`` and any final path segment named like
    the results folder token. Falls back to inserting ``output_folder`` next to
    the leaf directory when no token is found.
    """
    posix = fp.replace("\\", "/")
    token = f"/{results_token.strip('/')}/"
    if token in posix:
        return posix.replace(token, f"/{output_folder}/")
    # Fallback: replace the directory that contains Network (...) leaves' parent chain.
    # e.g. .../my_results/2C_10N/... → .../json_paths/2C_10N/...
    parts = Path(fp).parts
    # Prefer a segment that looks like a results root (contains C_N pattern next).
    for i, part in enumerate(parts):
        if i + 1 < len(parts) and re.match(r"\d+C_\d+N$", parts[i + 1]):
            new_parts = list(parts[:i]) + [output_folder] + list(parts[i + 1 :])
            return str(Path(*new_parts))
        if part in {"resultados", "resultados_ari"} or part == results_token:
            new_parts = list(parts[:i]) + [output_folder] + list(parts[i + 1 :])
            return str(Path(*new_parts))
    parent = Path(fp).parent
    return str(parent.parent / output_folder / parent.name / Path(fp).name)


def dump_all_json_paths(
    list_of_json_lists: list[list[str]],
    verbose: bool = False,
    output_folder: str = "json_paths",
    results_token: str = "resultados",
) -> list[str]:
    """Dump each path group into a text file under a sibling folder name."""
    tasks = []
    for json_list in list_of_json_lists:
        if not json_list:
            continue
        fp = json_list[0]
        matches = re.findall(r"Network \(0*(\d+)\)", fp)
        if not matches:
            continue
        network_seed = int(matches[0])
        output_file = _sibling_output_path(fp, results_token, output_folder)
        output_file_parts = output_file.replace("\\", "/").split("/")[:-1]
        output_file_parts[-1] = f"network_{network_seed:03d}.txt"
        output_file = "/".join(output_file_parts)
        tasks.append((json_list, output_file, verbose))
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        list(
            tqdm(
                executor.map(dump_json_paths_wrapper, tasks),
                total=len(tasks),
                desc="Dumping JSON paths",
            )
        )
    return [task[1] for task in tasks]


def _load_json_records(fp: str, ignore_partition_key: bool = True) -> list[dict]:
    """Load a result JSON that may be a single dict or a list of dicts."""
    with open(fp, "r", encoding="utf-8") as f:
        file_data = json.load(f)
    if isinstance(file_data, dict):
        records = [file_data]
    elif isinstance(file_data, list):
        records = file_data
    else:
        return []
    out = []
    for d in records:
        if not isinstance(d, dict):
            continue
        d = dict(d)
        if ignore_partition_key and "partition" in d:
            del d["partition"]
        out.append(d)
    return out


def dump_csv(
    txt_path: str,
    ignore_partition_key: bool = True,
    output_folder: str = "csv_results",
    paths_token: str = "json_paths",
) -> str:
    """Read a text file of JSON paths and dump combined rows to a gzipped CSV."""
    with open(txt_path, "r", encoding="utf-8") as f:
        file_paths = [fp.strip() for fp in f.readlines() if fp.strip()]
    data = []
    for fp in file_paths:
        data.extend(_load_json_records(fp, ignore_partition_key=ignore_partition_key))
    df = pd.DataFrame(data)
    csv_path = _sibling_output_path(txt_path, paths_token, output_folder).replace(
        ".txt", ".csv.gzip"
    )
    if csv_path.endswith(".txt.csv.gzip"):
        csv_path = csv_path[: -len(".txt.csv.gzip")] + ".csv.gzip"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False, compression="gzip")
    return csv_path


def dump_csv_helper(args: tuple[str, str]) -> str:
    txt_path, output_folder = args
    return dump_csv(txt_path, output_folder=output_folder)


def dump_all_csv(
    json_path: str | list[str], output_folder: str = "csv_results"
) -> list[str]:
    if isinstance(json_path, str):
        sorted_txt_paths = get_paths_sorted(json_path, extension=".txt")
    else:
        sorted_txt_paths = sort_files(json_path)
    args_list = [(fp, output_folder) for fp in sorted_txt_paths]
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        csv_paths = list(
            tqdm(
                executor.map(dump_csv_helper, args_list),
                total=len(sorted_txt_paths),
                desc="Dumping CSV files",
            )
        )
    return csv_paths


def read_csv_helper(fp: str) -> pd.DataFrame:
    return pd.read_csv(fp, compression="gzip")


def get_combined_dataframe(csv_path: str | list[str]) -> pd.DataFrame:
    if isinstance(csv_path, str):
        sorted_csv_paths = get_paths_sorted(csv_path, extension=".csv.gzip")
    else:
        sorted_csv_paths = sort_files(csv_path)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        df_list = list(
            tqdm(
                executor.map(read_csv_helper, sorted_csv_paths),
                total=len(sorted_csv_paths),
                desc="Combining CSV files",
            )
        )
    return pd.concat(df_list, ignore_index=True)


def load_experiment_data(
    results_folder: str,
    *,
    results_token: str | None = None,
    simple: bool = False,
) -> pd.DataFrame:
    """Full pipeline: index JSONs → path lists → CSVs → combined DataFrame.

    Parameters
    ----------
    simple :
        If True, load every JSON under ``results_folder`` directly into one
        DataFrame (fast path for smoke / small runs). Skips the intermediate
        json_paths / csv_results tree.
    results_token :
        Path segment to rewrite when placing json_paths/csv_results siblings
        (default: basename of ``results_folder``, usually ``resultados``).
    """
    if simple:
        sorted_json_paths = get_paths_sorted(results_folder)
        records: list[dict] = []
        for fp in tqdm(sorted_json_paths, desc="Loading JSON"):
            records.extend(_load_json_records(fp, ignore_partition_key=True))
        return pd.DataFrame(records)

    token = results_token or Path(results_folder.rstrip("/\\")).name or "resultados"
    sorted_json_paths = get_paths_sorted(results_folder)
    split_paths = split_json_paths(sorted_json_paths)
    txt_paths = dump_all_json_paths(split_paths, results_token=token)
    time.sleep(2)
    csv_paths = dump_all_csv(txt_paths)
    time.sleep(2)
    return get_combined_dataframe(csv_paths)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Load and combine disjoint experiment JSON results "
            "(V1020 partition_*.json lists or legacy Method.json files)."
        )
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default=str(DISJOINT_ARTIFACTS_DIR / "resultados"),
        help=(
            "Folder of raw JSON experiment results (default: "
            "repository artifacts/disjoint/resultados)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output gzipped CSV path (default: "
            "artifacts/disjoint/resultados.csv.gzip)"
        ),
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help=(
            "Load all JSONs directly into one CSV (recommended for smoke / "
            "small CLI runs; skips intermediate json_paths/csv_results trees)."
        ),
    )
    args = parser.parse_args(argv)

    results_folder = str(expand_path(args.results_folder))
    print("Loading experiment data from", results_folder)
    df = load_experiment_data(results_folder, simple=args.simple)
    output_path = str(expand_path(args.output)) if args.output else None
    if output_path is None:
        output_path = str(DISJOINT_ARTIFACTS_DIR / "resultados.csv.gzip")
    print("Saving data to", output_path)
    df.to_csv(output_path, index=False, compression="gzip")
    print(f"Done. rows={len(df)} cols={list(df.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
