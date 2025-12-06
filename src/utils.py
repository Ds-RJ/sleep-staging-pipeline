
import os
import numpy as np
import mne


def read_hyp_labels(hyp_path: str, valid_codes: set[int]) -> list[int]:
    """Read Sleep-EDF .hyp hypnogram labels (one uint8 per 30s epoch after 512-byte header).
    Returns raw codes; filters to valid codes while preserving order."""
    with open(hyp_path, "rb") as f:
        b = f.read()
    if len(b) <= 512:
        raise ValueError(f"Hyp file too short: {hyp_path}")
    payload = b[512:]
    labels_u8 = np.frombuffer(payload, dtype=np.uint8).tolist()
    labels = [code for code in labels_u8 if code in valid_codes]
    return labels


def find_pairs(data_dir: str) -> list[tuple[str, str]]:
    """Pair signal files (.rec/.edf) with hyp files (.hyp) by stem."""
    files = os.listdir(data_dir)
    sigs = [f for f in files if f.lower().endswith((".rec", ".edf"))]
    hyps = {os.path.splitext(f)[0]: f for f in files if f.lower().endswith(".hyp")]
    pairs = []
    for sig in sigs:
        stem = os.path.splitext(sig)[0]
        if stem in hyps:
            pairs.append((os.path.join(data_dir, sig), os.path.join(data_dir, hyps[stem])))
    return pairs


def read_raw_signal(sig_path: str) -> mne.io.BaseRaw:
    """Read .edf or .rec (EDF content) with MNE. If .rec causes trouble, rename temporarily."""
    if sig_path.lower().endswith(".rec"):
        tmp = sig_path[:-4] + ".edf"
        os.rename(sig_path, tmp)
        try:
            raw = mne.io.read_raw_edf(tmp, preload=True, verbose=False)
        finally:
            os.rename(tmp, sig_path)
        return raw
    else:
        raw = mne.io.read_raw_edf(sig_path, preload=True, verbose=False)
        return raw


def epoch_raw_30s(raw: mne.io.BaseRaw, epoch_len_sec: int = 30):
    """Epoch raw to contiguous 30s windows; drop remainder."""
    sfreq = float(raw.info["sfreq"])
    samples_per_epoch = int(round(epoch_len_sec * sfreq))
    total_samples = raw.n_times
    n_full_epochs = total_samples // samples_per_epoch
    data = raw.get_data()[:, :n_full_epochs * samples_per_epoch]
    data = data.reshape(data.shape[0], n_full_epochs, samples_per_epoch).transpose(1, 0, 2)
    return data, n_full_epochs, samples_per_epoch, sfreq


def aasm_from_rk(name: str) -> str:
    """Map R&K names to AASM classes."""
    if name in ["Stage 3", "Stage 4"]:
        return "N3"
    if name == "Stage 2":
        return "N2"
    if name == "Stage 1":
        return "N1"
    if name == "REM":
        return "REM"
    if name in ["Wake", "Awake"]:
        return "W"
    return "Unknown"
