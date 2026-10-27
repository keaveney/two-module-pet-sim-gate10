"""Shared optical-hit reduction: virtual 8x8 SiPM channels and sampled PE counts."""

from pathlib import Path

import numpy as np
import pandas as pd
import uproot

N_PIXELS = 8
PITCH_MM = 3.2
PIXEL_WIDTH_MM = 3.0
SIPM_THICKNESS_MM = 0.5
ROOT_KEYS = ["RunID", "EventID"]
CHANNEL_COLUMNS = [f"npe_{ix}_{iy}" for ix in range(N_PIXELS) for iy in range(N_PIXELS)]


def read_root(path: Path, branches: list[str]) -> pd.DataFrame:
    """Read branches with compatibility for uproot 4 and legacy uproot 3."""
    if not path.exists():
        raise FileNotFoundError(f"Missing ROOT input: {path}")
    with uproot.open(path) as root_file:
        try:
            tree_names = root_file.keys(cycle=False)
        except TypeError:
            tree_names = root_file.keys()
        if not tree_names:
            return pd.DataFrame(columns=branches)
        tree = root_file[tree_names[0]]
        available_raw = list(tree.keys())
        available = {name.decode() if isinstance(name, bytes) else name for name in available_raw}
        missing = set(branches).difference(available)
        if missing:
            raise RuntimeError(f"{path.name} is missing branches: {sorted(missing)}")
        # Older uproot expects byte branch names and returns byte column names.
        use_bytes = bool(available_raw and isinstance(available_raw[0], bytes))
        requested = [name.encode() if use_bytes else name for name in branches]
        try:
            arrays = tree.arrays(requested, library="np")
        except TypeError:  # uproot 3
            arrays = tree.arrays(requested)
        frame = pd.DataFrame(arrays)
        frame.columns = [name.decode() if isinstance(name, bytes) else name for name in frame.columns]
        for name in frame.columns:
            if frame[name].dtype == object:
                frame[name] = frame[name].map(
                    lambda value: value.decode() if isinstance(value, bytes) else value
                )
        return frame


def pixel_indices(x_mm: np.ndarray, y_mm: np.ndarray):
    """Map local SiPM coordinates to the 8 x 8 active virtual channels."""
    offset = (N_PIXELS - 1) * PITCH_MM / 2
    ix = np.rint((x_mm + offset) / PITCH_MM).astype(int)
    iy = np.rint((y_mm + offset) / PITCH_MM).astype(int)
    valid = (ix >= 0) & (ix < N_PIXELS) & (iy >= 0) & (iy < N_PIXELS)
    x_centre = ix * PITCH_MM - offset
    y_centre = iy * PITCH_MM - offset
    valid &= np.abs(x_mm - x_centre) <= PIXEL_WIDTH_MM / 2 + 1e-6
    valid &= np.abs(y_mm - y_centre) <= PIXEL_WIDTH_MM / 2 + 1e-6
    return ix, iy, valid


def reduce_module(input_dir: Path, module: str, pde: float, seed: int) -> pd.DataFrame:
    """Reduce optical arrivals in one module to one detector-template row/event."""
    hits = read_root(
        input_dir / f"sipm_hits_{module}.root",
        ROOT_KEYS + ["PrePositionLocal_X", "PrePositionLocal_Y", "PrePositionLocal_Z", "GlobalTime"],
    )
    if hits.empty:
        return pd.DataFrame()

    front = np.isclose(hits["PrePositionLocal_Z"], -SIPM_THICKNESS_MM / 2, atol=1e-5)
    hits = hits.loc[front].copy()
    ix, iy, active = pixel_indices(hits["PrePositionLocal_X"].to_numpy(), hits["PrePositionLocal_Y"].to_numpy())
    hits = hits.loc[active].copy()
    hits["ix"] = ix[active]
    hits["iy"] = iy[active]
    if hits.empty:
        return pd.DataFrame()

    # One Bernoulli draw per arriving photon gives both a binomial PE count and
    # the time of the first detected PE without modelling an electronics pulse.
    rng = np.random.default_rng(seed)
    detected = hits.loc[rng.random(len(hits)) < pde].copy()
    if detected.empty:
        return pd.DataFrame()

    optical = hits.groupby(ROOT_KEYS + ["ix", "iy"], as_index=False).size().rename(columns={"size": "n_optical"})
    pe = detected.groupby(ROOT_KEYS + ["ix", "iy"], as_index=False).size().rename(columns={"size": "npe"})
    channels = optical.merge(pe, on=ROOT_KEYS + ["ix", "iy"], how="left")
    channels["npe"] = channels["npe"].fillna(0).astype(int)

    totals = channels.groupby(ROOT_KEYS, as_index=False)[["n_optical", "npe"]].sum()
    totals = totals.rename(columns={"n_optical": "n_optical_total", "npe": "npe_total"})
    first_pe = detected.groupby(ROOT_KEYS, as_index=False)["GlobalTime"].min()
    first_pe = first_pe.rename(columns={"GlobalTime": "first_pe_time_ns"})

    maximum = (channels.sort_values(ROOT_KEYS + ["npe", "n_optical", "ix", "iy"],
                                    ascending=[True, True, False, False, True, True])
                       .drop_duplicates(ROOT_KEYS)
                       [[*ROOT_KEYS, "ix", "iy", "npe", "n_optical"]]
                       .rename(columns={"ix": "max_ix", "iy": "max_iy", "npe": "max_npe",
                                        "n_optical": "max_n_optical"}))

    image = channels.pivot(index=ROOT_KEYS, columns=["ix", "iy"], values="npe")
    image = image.reindex(columns=pd.MultiIndex.from_product([range(N_PIXELS), range(N_PIXELS)]), fill_value=0)
    image = image.fillna(0).astype(int)
    image.columns = CHANNEL_COLUMNS
    image = image.reset_index()
    return totals.merge(first_pe, on=ROOT_KEYS).merge(maximum, on=ROOT_KEYS).merge(image, on=ROOT_KEYS)

