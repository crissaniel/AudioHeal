import argparse
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.io import wavfile
except ImportError as e:
    raise SystemExit("Missing dependency: scipy. Install with: pip install scipy") from e


# =========================
# Utilities
# =========================

def to_float_audio(x: np.ndarray) -> Tuple[np.ndarray, np.dtype]:
    """
    Convert int PCM audio to float32 in [-1, 1] (approx), keep dtype for writing back.
    Supports int16, int32, uint8, float32/float64.
    """
    orig_dtype = x.dtype

    if np.issubdtype(orig_dtype, np.floating):
        xf = x.astype(np.float32)
        # assume already -1..1 (common), but don't force
        return xf, orig_dtype

    if orig_dtype == np.uint8:
        # unsigned 8-bit PCM: 0..255 where 128 is ~0
        xf = (x.astype(np.float32) - 128.0) / 128.0
        return xf, orig_dtype

    if orig_dtype == np.int16:
        xf = x.astype(np.float32) / 32768.0
        return xf, orig_dtype

    if orig_dtype == np.int32:
        xf = x.astype(np.float32) / 2147483648.0
        return xf, orig_dtype

    # fallback
    xf = x.astype(np.float32)
    return xf, orig_dtype


def from_float_audio(xf: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Convert float32 audio back to original dtype, clipping to valid range.
    """
    xf = np.asarray(xf)

    if np.issubdtype(dtype, np.floating):
        return xf.astype(dtype)

    if dtype == np.uint8:
        y = np.clip(xf, -1.0, 1.0)
        y = (y * 128.0 + 128.0).round()
        return y.astype(np.uint8)

    if dtype == np.int16:
        y = np.clip(xf, -1.0, 0.9999695)
        y = (y * 32768.0).round()
        return y.astype(np.int16)

    if dtype == np.int32:
        y = np.clip(xf, -1.0, 0.999999999)
        y = (y * 2147483648.0).round()
        return y.astype(np.int32)

    # fallback
    return xf.astype(dtype)


def time_to_index(t: float, fs: int) -> int:
    return int(round(t * fs))


# =========================
# Interpolation Methods
# =========================

def lagrange_interpolate(x_points: np.ndarray, y_points: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """
    Lagrange polynomial interpolation.
    x_points: (m,)
    y_points: (m,)
    x_eval: (k,)
    returns y_eval: (k,)
    """
    x_points = np.asarray(x_points, dtype=np.float64)
    y_points = np.asarray(y_points, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)

    m = len(x_points)
    y_eval = np.zeros_like(x_eval, dtype=np.float64)

    for i in range(m):
        # basis polynomial Li(x)
        xi = x_points[i]
        denom = 1.0
        for j in range(m):
            if j == i:
                continue
            denom *= (xi - x_points[j])

        # avoid division by 0 if duplicates
        if denom == 0:
            continue

        numer = np.ones_like(x_eval, dtype=np.float64)
        for j in range(m):
            if j == i:
                continue
            numer *= (x_eval - x_points[j])

        y_eval += y_points[i] * (numer / denom)

    return y_eval


def newton_divided_differences_coeffs(x_points: np.ndarray, y_points: np.ndarray) -> np.ndarray:
    """
    Compute Newton divided difference coefficients.
    Returns coeffs c where:
    P(x) = c0 + c1(x-x0) + c2(x-x0)(x-x1) + ...
    """
    x = np.asarray(x_points, dtype=np.float64)
    coef = np.asarray(y_points, dtype=np.float64).copy()

    n = len(x)
    # In-place divided differences
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1:n - 1]) / (x[j:n] - x[0:n - j])

    return coef


def newton_evaluate(x_points: np.ndarray, coeffs: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """
    Evaluate Newton form polynomial at x_eval using nested multiplication.
    """
    x_points = np.asarray(x_points, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)

    n = len(coeffs)
    y = np.full_like(x_eval, coeffs[n - 1], dtype=np.float64)
    for k in range(n - 2, -1, -1):
        y = y * (x_eval - x_points[k]) + coeffs[k]
    return y


def linear_interpolate(x_points: np.ndarray, y_points: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """
    Linear interpolation using numpy interp (requires x_points sorted).
    """
    return np.interp(x_eval, x_points, y_points)


# =========================
# Gap Healing
# =========================

@dataclass
class Gap:
    start_idx: int
    end_idx: int  # exclusive


def simulate_damage(audio: np.ndarray, gap: Gap, fill_value: float = 0.0) -> np.ndarray:
    damaged = audio.copy()
    damaged[gap.start_idx:gap.end_idx, ...] = fill_value
    return damaged


def pick_context_points(
    y: np.ndarray, gap: Gap, fs: int, method_order: int, context_ms: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Choose interpolation anchor points around the gap.
    - method_order = polynomial degree (e.g., 3 for cubic)
    - points needed = degree+1
    - we pick half points before and half after the gap (as balanced as possible)
    """
    points_needed = method_order + 1
    half = points_needed // 2
    # If odd number of points, take one extra on the right
    left_needed = half
    right_needed = points_needed - left_needed

    context_samples = max(1, int(round((context_ms / 1000.0) * fs)))

    # Candidate indices before gap
    left_start = max(0, gap.start_idx - context_samples)
    left_region = np.arange(left_start, gap.start_idx)
    # Candidate indices after gap
    right_end = min(len(y), gap.end_idx + context_samples)
    right_region = np.arange(gap.end_idx, right_end)

    if len(left_region) < left_needed or len(right_region) < right_needed:
        raise ValueError(
            f"Not enough context samples. Need left={left_needed}, right={right_needed}, "
            f"but got left={len(left_region)}, right={len(right_region)}. Increase context_ms."
        )

    # Pick points evenly from regions
    left_idx = np.linspace(0, len(left_region) - 1, left_needed).round().astype(int)
    right_idx = np.linspace(0, len(right_region) - 1, right_needed).round().astype(int)

    anchor_idx = np.concatenate([left_region[left_idx], right_region[right_idx]])
    anchor_idx = np.unique(anchor_idx)  # ensure no duplicates

    # If unique reduced count, pad by selecting nearest unused
    while len(anchor_idx) < points_needed:
        # prefer adding from left then right
        candidates = np.concatenate([left_region[::-1], right_region])
        for c in candidates:
            if c not in anchor_idx:
                anchor_idx = np.append(anchor_idx, c)
                break
        anchor_idx = np.unique(anchor_idx)

    anchor_idx = np.sort(anchor_idx)
    x_points = anchor_idx.astype(np.float64)
    y_points = y[anchor_idx].astype(np.float64)
    return x_points, y_points


def heal_gap_1d(
    y: np.ndarray,
    gap: Gap,
    fs: int,
    method: str,
    order: int,
    context_ms: float
) -> np.ndarray:
    """
    Heal one channel (1D signal).
    """
    y_healed = y.copy()
    x_eval = np.arange(gap.start_idx, gap.end_idx, dtype=np.float64)

    if method == "linear":
        # for linear, force order=1 and just use two anchor points (one before, one after)
        x_points, y_points = pick_context_points(y, gap, fs, method_order=1, context_ms=context_ms)
        y_fill = linear_interpolate(x_points, y_points, x_eval)

    elif method == "lagrange":
        x_points, y_points = pick_context_points(y, gap, fs, method_order=order, context_ms=context_ms)
        y_fill = lagrange_interpolate(x_points, y_points, x_eval)

    elif method == "newton":
        x_points, y_points = pick_context_points(y, gap, fs, method_order=order, context_ms=context_ms)
        coeffs = newton_divided_differences_coeffs(x_points, y_points)
        y_fill = newton_evaluate(x_points, coeffs, x_eval)

    else:
        raise ValueError("Unknown method. Use: linear, lagrange, newton")

    # Write fill, clip to safe float range
    y_healed[gap.start_idx:gap.end_idx] = np.clip(y_fill, -1.0, 1.0).astype(np.float32)
    return y_healed


def heal_audio(
    audio: np.ndarray,
    gap: Gap,
    fs: int,
    method: str,
    order: int,
    context_ms: float
) -> np.ndarray:
    """
    Heal mono or multi-channel audio.
    audio: float32 (N,) or (N, C)
    """
    if audio.ndim == 1:
        return heal_gap_1d(audio, gap, fs, method, order, context_ms)

    # multi-channel
    out = audio.copy()
    for ch in range(audio.shape[1]):
        out[:, ch] = heal_gap_1d(audio[:, ch], gap, fs, method, order, context_ms)
    return out


# =========================
# Visualization
# =========================

def plot_signals(fs: int, original: np.ndarray, damaged: np.ndarray, healed: np.ndarray, gap: Gap, title: str):
    n = len(original)
    t = np.arange(n) / fs

    # If multi-channel, show channel 0 for clarity
    def ch0(x):
        return x if x.ndim == 1 else x[:, 0]

    o = ch0(original)
    d = ch0(damaged)
    h = ch0(healed)

    plt.figure(figsize=(12, 5))
    plt.plot(t, o, label="Original", linewidth=1)
    plt.plot(t, d, label="Damaged", linewidth=1)
    plt.plot(t, h, label="Healed", linewidth=1)
    plt.xlim(
    gap.start_idx / fs - 0.05,
    gap.end_idx / fs + 0.05
    )


    # Shade gap
    t0 = gap.start_idx / fs
    t1 = gap.end_idx / fs
    plt.axvspan(t0, t1, alpha=0.2, label="Gap")

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# CLI Entry Point
# =========================

def main():
    parser = argparse.ArgumentParser(description="AudioHeal - heal corrupted audio gaps with interpolation.")
    parser.add_argument("input_wav", help="Path to input WAV file.")
    parser.add_argument("--output_wav", default="healed_output.wav", help="Path to output WAV file.")
    parser.add_argument("--gap_start", type=float, required=True, help="Gap start time in seconds.")
    parser.add_argument("--gap_end", type=float, required=True, help="Gap end time in seconds.")
    parser.add_argument("--simulate", action="store_true", help="Simulate damage by zeroing the gap first.")
    parser.add_argument("--method", choices=["linear", "lagrange", "newton"], default="lagrange")
    parser.add_argument("--order", type=int, default=3, help="Polynomial order (degree). e.g., 3=cubic.")
    parser.add_argument("--context_ms", type=float, default=10.0, help="Context window (ms) around gap to pick points.")
    parser.add_argument("--plot", action="store_true", help="Plot Original/Damaged/Healed waveforms.")
    args = parser.parse_args()

    fs, audio_raw = wavfile.read(args.input_wav)
    audio_f, orig_dtype = to_float_audio(audio_raw)

    start_idx = time_to_index(args.gap_start, fs)
    end_idx = time_to_index(args.gap_end, fs)

    if start_idx < 0 or end_idx <= start_idx or end_idx > len(audio_f):
        raise SystemExit("Invalid gap range. Check --gap_start and --gap_end.")

    gap = Gap(start_idx=start_idx, end_idx=end_idx)

    original = audio_f
    damaged = simulate_damage(audio_f, gap, fill_value=0.0) if args.simulate else audio_f.copy()

    damaged_out = from_float_audio(damaged, orig_dtype)
    wavfile.write("damaged.wav", fs, damaged_out)

    # For non-simulated mode, we still overwrite the gap during healing
    working = damaged.copy()

    healed = heal_audio(
        audio=working,
        gap=gap,
        fs=fs,
        method=args.method,
        order=args.order,
        context_ms=args.context_ms
    )

    # Write output
    healed_out = from_float_audio(healed, orig_dtype)
    wavfile.write(args.output_wav, fs, healed_out)

    print(f"Saved healed audio to: {args.output_wav}")
    print(f"Method={args.method}, Order={args.order}, Gap=[{args.gap_start}s, {args.gap_end}s], Context={args.context_ms}ms")

    if args.plot:
        plot_signals(
            fs=fs,
            original=original,
            damaged=damaged,
            healed=healed,
            gap=gap,
            title=f"AudioHeal ({args.method}, order={args.order})"
        )


if __name__ == "__main__":
    main()
