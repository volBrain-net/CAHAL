bins = [1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 10.5]
bin_labels = [1, 2, 3, 4, 5, 6, 7]

anisotropy_bins = [0.01, 0.33, 0.66, 1.0]
anisotropy_labels = [1, 2, 3]  # 1=high anisotropy, 2=medium, 3=high isotropy


def get_anisotropy_bin(resolutions):
    min_res, max_res = min(resolutions), max(resolutions)
    anisotropy = min_res / max_res if max_res > 0 else 1.0
    # Right-closed bins, matching the pd.cut convention used for the real database in data.py.
    if anisotropy <= 0.33:
        return 1
    elif anisotropy <= 0.66:
        return 2
    else:
        return 3
