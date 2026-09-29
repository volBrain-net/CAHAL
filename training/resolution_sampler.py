import random
import numpy as np

from clusters import bins, get_anisotropy_bin


def sample_resolutions_two_equal(vol_bin, aniso_bin, per_axis_min=1.0, per_axis_max=10.0, max_resample=20, rng=None):
    """
    Sample a resolution triplet (r, r, s) whose product falls in the vol_bin range
    and whose anisotropy (min/max) falls in the aniso_bin range.
    """
    if rng is None:
        rng = np.random.default_rng()

    L, U = bins[vol_bin - 1], bins[vol_bin]
    a_low, a_high = {1: (0.01, 0.33), 2: (0.33, 0.66), 3: (0.66, 1.0)}[aniso_bin]

    for _ in range(max_resample):
        P = rng.uniform(L, U)
        a = rng.uniform(a_low, a_high)

        # (r, r, s): product r^2*s = P, anisotropy min/max = a
        if rng.choice([1, 2]) == 1:  # r <= s
            r = (P * a) ** (1 / 3)
            s = r / a
        else:  # r >= s
            r = (P / a) ** (1 / 3)
            s = a * r

        if not (per_axis_min <= r <= per_axis_max and per_axis_min <= s <= per_axis_max):
            continue

        product_error = abs(r * r * s - P) / max(P, 1e-10)
        aniso_check = min(r, s) / max(r, s)
        if product_error < 1e-6 and a_low - 1e-6 <= aniso_check <= a_high + 1e-6:
            triplet = [r, r, s]
            rng.shuffle(triplet)
            return [float(x) for x in triplet]

    raise RuntimeError(f"Failed to sample resolutions for vol_bin={vol_bin}, aniso_bin={aniso_bin}")


def get_resolutions_from_database(resolutions_df, vol_bin, aniso_bin, max_tries=20):
    """Draw an observed resolution triplet matching both the volumetric and anisotropy cluster."""
    matches = resolutions_df[(resolutions_df["cluster"] == vol_bin) &
                              (resolutions_df["anisotropy_cluster"] == aniso_bin)]
    if matches.empty:
        raise RuntimeError(f"No database samples for vol_bin={vol_bin}, aniso_bin={aniso_bin}")

    for _ in range(max_tries):
        sample = matches.sample(n=1)
        resolutions = [float(sample["Resolution_X"].values[0]),
                        float(sample["Resolution_Y"].values[0]),
                        float(sample["Resolution_Z"].values[0])]
        if all(r > 0 for r in resolutions):
            return resolutions

    raise RuntimeError(f"Could not draw a valid database sample for vol_bin={vol_bin}, aniso_bin={aniso_bin}")


def sample_resolutions(vol_bin, aniso_bin, resolutions_df, max_attempts=5):
    """
    Sample a resolution triplet for a given (vol_bin, aniso_bin):
    """
    free_aniso = aniso_bin == -1
    if free_aniso:
        aniso_bin = np.random.choice([1, 2, 3])
    if vol_bin == -1:
        vol_bin = np.random.choice([1, 2, 3, 4, 5, 6, 7])

    lower_bound, upper_bound = bins[vol_bin - 1], bins[vol_bin]

    for _ in range(max_attempts):
        try:
            if np.random.rand() < 0.5:
                resolutions = get_resolutions_from_database(resolutions_df, vol_bin, aniso_bin)
            else:
                try:
                    resolutions = sample_resolutions_two_equal(vol_bin, aniso_bin)
                    random.shuffle(resolutions)
                except RuntimeError:
                    resolutions = get_resolutions_from_database(resolutions_df, vol_bin, aniso_bin)
        except RuntimeError:
            continue  # this attempt's source had no valid samples; try again

        if aniso_bin == 1 and np.random.rand() < 0.33:
            resolutions = [1.0, 1.0, np.random.uniform(lower_bound, upper_bound)]
            random.shuffle(resolutions)

        if get_anisotropy_bin(resolutions) == aniso_bin:
            return resolutions

    raise RuntimeError(f"Failed to generate resolutions matching vol_bin={vol_bin}, aniso_bin={aniso_bin}")
