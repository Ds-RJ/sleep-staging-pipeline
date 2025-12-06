
import os
import argparse
import yaml
import pickle
import numpy as np
import pandas as pd
import scipy.signal as sps


def welch_bandpowers(x, sfreq, bands, fmin=0.5, fmax=50, nperseg=512, noverlap=256, window="hann"):
    f, psd = sps.welch(x, fs=sfreq, window=window, nperseg=nperseg, noverlap=noverlap)
    mask_total = (f >= fmin) & (f <= fmax)
    total_power = np.trapz(psd[mask_total], f[mask_total])
    bp_abs, bp_rel = {}, {}
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f <= hi)
        band_power = np.trapz(psd[m], f[m])
        bp_abs[name] = float(band_power)
        bp_rel[name] = float(band_power / total_power) if total_power > 0 else 0.0
    # spectral entropy
    p = psd[mask_total]
    if np.sum(p) > 0:
        p_norm = p / np.sum(p)
        p_norm = np.clip(p_norm, 1e-12, None)
        ent = float(-np.sum(p_norm * np.log2(p_norm)))
    else:
        ent = 0.0
    return bp_abs, bp_rel, ent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_pickle", type=str, required=True, help="Synced epochs pickle")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--out_parquet", type=str, default="")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    with open(args.in_pickle, "rb") as f:
        synced = pickle.load(f)

    bands = {k: tuple(v) for k, v in cfg["bands_hz"].items()}
    channels_to_use = cfg["channels_to_use"]

    rows = []
    for file_key, d in synced.items():
        sfreq = float(d["sfreq"])
        X = d["epochs_data"]         # (n_epochs, n_channels, samples)
        ch_names = d["channel_names"]
        labels = d["label_names"]
        # channel indices to use
        idxs = [ch_names.index(ch) for ch in channels_to_use if ch in ch_names]
        if not idxs:
            idxs = [0]

        for i_epoch in range(X.shape[0]):
            feats = {
                "file": file_key,
                "epoch_idx": i_epoch,
                "sfreq": sfreq,
                "duration_sec": d["samples_per_epoch"] / sfreq,
                "label": labels[i_epoch]
            }
            agg_abs = {b: [] for b in bands.keys()}
            agg_rel = {b: [] for b in bands.keys()}
            entropies = []

            for ci in idxs:
                chunk = X[i_epoch, ci, :]
                bp_abs, bp_rel, ent = welch_bandpowers(chunk, sfreq, bands)
                entropies.append(ent)
                for b in bands.keys():
                    agg_abs[b].append(bp_abs[b])
                    agg_rel[b].append(bp_rel[b])

            feats["psd_entropy_mean"] = float(np.mean(entropies))
            for b in bands.keys():
                feats[f"{b}_power_abs_mean"] = float(np.mean(agg_abs[b]))
                feats[f"{b}_power_rel_mean"] = float(np.mean(agg_rel[b]))
            rows.append(feats)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    if args.out_parquet:
        df.to_parquet(args.out_parquet, index=False)
    print(f"Saved features -> {args.out_csv}")
    if args.out_parquet:
        print(f"Saved features (parquet) -> {args.out_parquet}")


if __name__ == "__main__":
    main()
