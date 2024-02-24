import os
import unittest
from typing import Dict, Union

import numpy as np

from current_reconstruction import Image, reconstruct_current

data_dir = os.path.join(os.path.dirname(__file__), os.pardir, "sample_data")


def load_dataset(npz_path: str) -> Dict[str, Union[np.ndarray, float, str]]:
    with np.load(npz_path, "r") as f:
        dataset = dict(f)
    for key, value in dataset.items():
        if value.ndim == 0:  # scalar value
            dataset[key] = value.item()
    return dataset


def load_psf_data() -> Dict[str, Union[np.ndarray, float, str]]:
    return load_dataset(os.path.join(data_dir, "psf.npz"))


def load_mag_data() -> Dict[str, Union[np.ndarray, float, str]]:
    return load_dataset(os.path.join(data_dir, "mag.npz"))


def reconstruct_and_plot_current(
    mag: Image,
    psf: Image,
    standoff: float = 1.0,
    kx_max: float = 1.0,
    ky_max: float = 1.0,
    normalize: bool = True,
    quiver_every: int = 5,
    quiver_scale: float = 15.0,
):
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    jx, jy = reconstruct_current(mag, standoff, psf=psf, kx_max=kx_max, ky_max=ky_max)

    j = np.sqrt(jx**2 + jy**2)
    jmax = np.max(np.abs(j))

    if normalize:
        jx /= jmax
        jy /= jmax
        j /= jmax
        vmin, vmax = -1, 1
    else:
        vmin, vmax = -jmax, jmax

    fig, axes = plt.subplots(3, 2, figsize=(8, 8), constrained_layout=True)

    xs = mag.xs
    ys = mag.ys

    cbarkw = dict(pad=0.02, shrink=0.9)

    ax = axes[0, 0]
    psf_vmax = np.max(np.abs(psf.data))
    im = ax.pcolormesh(
        psf.xs, psf.ys, psf.data, cmap="coolwarm", vmin=-psf_vmax, vmax=psf_vmax
    )
    cbar = fig.colorbar(im, ax=ax, location="top", pad=0.01, shrink=0.55)
    cbar.set_label("Sensor PSF [m$\\Phi_0$]")

    ax = axes[0, 1]
    current = 1e-6  # A
    mPhi_0_per_Phi_0 = 1e3
    im = ax.pcolormesh(xs, ys, mag.data * current * mPhi_0_per_Phi_0, cmap="cividis")
    cbar = fig.colorbar(im, ax=ax, **cbarkw)
    cbar.set_label("Magnetic flux [m$\\Phi_0$]")
    ax.set_title("AC Magnetometry")

    for ax, zs, label in zip(axes[1], [jx, jy], ["j_x", "j_y"]):
        im = ax.pcolormesh(xs, ys, zs, cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(f"${label}$")
    cbar = fig.colorbar(im, ax=axes[1], **cbarkw)
    cbar.set_label(f"2D current density ${label}$ [arb.]")

    for ax in axes[2]:
        im = ax.pcolormesh(xs, ys, j, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title("Current density")
    cbar = fig.colorbar(im, ax=axes[2], **cbarkw)
    cbar.set_label("| $\\vec{j}$ | [arb.]")

    ax = axes[2, 1]
    ix = slice(None, None, quiver_every)
    ax.quiver(
        xs[ix],
        ys[ix],
        jx[ix, ix],
        jy[ix, ix],
        color="w",
        scale=quiver_scale,
        alpha=0.75,
    )

    for ax, color, scale in zip(axes.flat, "kkkkww", [4] + [40] * 5):
        scalebar = AnchoredSizeBar(
            transform=ax.transData,
            size=scale,
            size_vertical=scale / 10,
            loc="lower left",
            label=f"{scale} $\\mu$m",
            color=color,
            frameon=False,
            label_top=True,
            pad=0.15,
            fontproperties=fm.FontProperties(size=14),
        )
        ax.add_artist(scalebar)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    return jx, jy, (fig, axes)


class TestCurrentReconstruction(unittest.TestCase):
    def test_basic(self):
        psf_data = load_psf_data()
        mag_data = load_mag_data()

        psf = Image(psf_data["xs"], psf_data["ys"], psf_data["data"])
        mag = Image(mag_data["xs"], mag_data["ys"], mag_data["data"])
        standoff = mag_data["standoff"]

        kx_max, ky_max = 1.0, 1.0

        jx, jy = reconstruct_current(mag, standoff, psf=psf, kx_max=kx_max, ky_max=ky_max)

        self.assertEqual(mag.data.shape, jx.shape)
        self.assertEqual(mag.data.shape, jy.shape)

    def test_reconstruct_and_plot_current(self):
        psf_data = load_psf_data()
        mag_data = load_mag_data()
        psf = Image(psf_data["xs"], psf_data["ys"], psf_data["data"])
        mag = Image(mag_data["xs"], mag_data["ys"], mag_data["data"])
        standoff = mag_data["standoff"]
        kx_max, ky_max = 1.0, 1.0
        jx, jy, (fig, axes) = reconstruct_and_plot_current(
            mag,
            psf,
            standoff,
            kx_max=kx_max,
            ky_max=ky_max,
        )
        fig.set_facecolor("w")
        # fig.savefig(
        #     "./sample_data/current_reconstruction.png", dpi=300, bbox_inches="tight"
        # )


if __name__ == "__main__":
    unittest.main()
