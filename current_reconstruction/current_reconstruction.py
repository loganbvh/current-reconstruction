"""This module implements a Fourier transform inversion of the 2D Biot-Savart law
to reconstruct a 2D current density distribution from an image of the out-of-plane
magnet field B_z, or a magnetic flux signal given by B_z convolved with a magnetic
sensor's point-spread function or imaging kernel.

The method is based on "Using a magnetometer to image a two‐dimensional current distribution",
J. Appl. Phys. 65, 361–372 (1989) https://doi.org/10.1063/1.342549. A free PDF is available at
https://www.vanderbilt.edu/lsp/documents/jap-roth-using-89.pdf
"""


from typing import Optional, Tuple

import numpy as np
import scipy
from scipy.constants import mu_0


def hanning_2D(kx: np.ndarray, ky: np.ndarray, kx_max: float, ky_max: float):
    """2D Hanning window."""
    # See Eq. 18 in J. Appl. Phys. 65, 361–372 (1989), https://doi.org/10.1063/1.342549
    Kx, Ky = np.meshgrid(kx, ky)
    K = np.sqrt((Kx / kx_max) ** 2 + (Ky / ky_max) ** 2)
    H = 0.5 * (1 + np.cos(np.pi * K))
    H[K > 1] = 0
    return H


class Image:
    """An image on a rectangular grid.

    Args:
        xs: The x coordinates of the image axes, shape (m, )
        ys: The y coordinates of the image axes, shape (n, )
        zs: The z values of the image, shape (n, m)
    """

    def __init__(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray):
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        zs = np.asarray(zs)

        if len(xs) != zs.shape[1] or len(ys) != zs.shape[0]:
            raise ValueError(
                f"unexpected shape: {xs.shape = }, {ys.shape = }, {zs.shape = }"
            )

        # Ensure xs and ys are evenly spaced
        dx = np.diff(xs)
        if not np.allclose(dx, dx[0]):
            raise ValueError("xs must be evenly space")

        dy = np.diff(ys)
        if not np.allclose(dy, dy[0]):
            raise ValueError("ys must be evenly space")

        # Ensure xs and ys are in increasing order
        dx = dx[0]
        dy = dy[0]
        if dx < 0:
            xs = xs[::-1]
            zs = np.fliplr(zs)
            dx *= -1
        if dy < 0:
            ys = ys[::-1]
            zs = np.flipud(zs)
            dy *= -1

        # Ensure an odd number of pixels along each axis
        if not len(xs) % 2:
            print("truncating image to an odd number of columns")
            xs = xs[:-1]
            zs = zs[:, :-1]
        if not len(ys) % 2:
            print("truncating image to an odd number of rows")
            ys = ys[:-1]
            zs = zs[:-1, :]

        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.dx = dx
        self.dy = dy

    def pad(self, x_pad: int, y_pad: int, mode: str = "linear_ramp") -> "Image":
        """Pad the image symmetrically by a given number of pixels.

        Args:
            x_pad: The number of pixels (columns) to pad on the left and right
            y_pad: The number of pixels (rows) to pad on the top and bottom
            mode: The padding mode; see documentation for numpy.pad

        Returns:
            A new padded image
        """
        zs = np.pad(self.zs, ((y_pad, y_pad), (x_pad, x_pad)), mode=mode)

        x0 = self.xs[0] - x_pad * self.dx
        x1 = self.xs[-1] + x_pad * self.dx
        xs = np.linspace(x0, x1, zs.shape[1])

        y0 = self.ys[0] - y_pad * self.dy
        y1 = self.ys[-1] + y_pad * self.dy
        ys = np.linspace(y0, y1, zs.shape[0])

        return Image(xs, ys, zs)

    def pad_to_match(self, other: "Image", mode: str = "constant") -> "Image":
        """Pad the image to match the physical dimensions of another image.

        Args:
            other: The image whose dimensions self will be padded to match
            mode: The padding mode; see documentation for numpy.pad

        Returns:
            A new padded image
        """
        diff_x = other.dx * len(other.xs) - self.dx * len(self.xs)
        diff_y = other.dy * len(other.ys) - self.dy * len(self.ys)

        x_pad = int(diff_x / self.dx / 2)
        y_pad = int(diff_y / self.dy / 2)

        if x_pad < 0 or y_pad < 0:
            raise ValueError("Cannot pad to match a smaller image.")

        return self.pad(x_pad, y_pad, mode=mode)

    def resample(self, Nx: int, Ny: int) -> "Image":
        """Interpolate the image to a given number of pixels along each axis.

        Args:
            Nx: The target number of columns
            Ny: The target number of rows

        Returns:
            A new resampled image
        """
        new_xs = np.linspace(self.xs.min(), self.xs.max(), Nx)
        new_ys = np.linspace(self.ys.min(), self.ys.max(), Ny)
        interp = scipy.interpolate.RectBivariateSpline(self.xs, self.ys, self.zs.T)
        return Image(new_xs, new_ys, interp(new_xs, new_ys).T)

    @staticmethod
    def ones_like(other: "Image") -> "Image":
        """Make and image of all ones with the same dimension as ``other``"""
        return Image(other.xs.copy(), other.ys.copy(), np.ones_like(other.zs))


def reconstruct_current(
    mag: Image,
    z0: float,
    psf: Optional[Image] = None,
    x_pad: Optional[int] = None,
    y_pad: Optional[int] = None,
    pad_mode: str = "linear_ramp",
    kx_max: float = 1.5,
    ky_max: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs a Fourier transform inversion of the 2D Biot-Savart law
    given a magnetometry image and a sensor point spread function.

    Args:
        mag: The magnetometry image
        z0: The sensor-sample standoff distance
        psf: The sensor point spread function. If none is given, it is assumed
            that mag.zs represents the z component of the magnetic field
            rather than a magnetic flux.
        x_pad: The number of columns by which to pad the
            magnetometry image prior to FFT. Default: int(len(mag.xs) / 2)
        x_pad: The number of columns by which to pad the
            magnetometry image prior to FFT. Default: int(len(mag.ys) / 2)
        pad_mode: The padding mode; see documentation for numpy.pad
        kx_max: kx cutoff for the Hanning window.
            Smaller values filter high spatial frequency components.
        ky_max: ky cutoff for the Hanning window.
            Smaller values filter high spatial frequency components.

    Returns:
        The x and y components of the sheet current density,
        both with the same shape as the magnetometry image.
    """

    if x_pad is None:
        x_pad = int(len(mag.xs) / 2)
    if y_pad is None:
        y_pad = int(len(mag.ys) / 2)

    # Fourier transform mag and PSF
    mag = mag.pad(x_pad=x_pad, y_pad=y_pad, mode=pad_mode)
    mag_k = np.fft.fftshift(np.fft.fft2(mag.zs))

    if psf is None:
        psf_k = 1.0
    else:
        psf = psf.pad_to_match(mag).resample(len(mag.xs), len(mag.ys))
        psf_k = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf.zs)))

    mag_k = np.fft.fftshift(np.fft.fft2(mag.zs))
    psf_k = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf.zs)))

    # Construct k-space coordinates
    dx, dy = mag.dx, mag.dy
    kx = np.linspace(-np.pi / dx, np.pi / dx, mag.zs.shape[1], endpoint=False)
    ky = np.linspace(-np.pi / dy, np.pi / dy, mag.zs.shape[0], endpoint=False)
    Kx, Ky = np.meshgrid(kx, ky)
    K = np.sqrt(Kx**2 + Ky**2)

    # Make Hanning filter
    H = hanning_2D(kx, ky, kx_max, ky_max)

    # Evaluate jx and jy in k-space
    factor = 2 * 1j * mag_k * H / (mu_0 * np.exp(-K * z0) * K * psf_k)
    jx_k = -Ky * factor
    jy_k = +Kx * factor

    # Evaluate jx and jy in real space
    jx = np.fft.ifft2(np.fft.ifftshift(jx_k))
    jy = np.fft.ifft2(np.fft.ifftshift(jy_k))

    # Crop back to original size
    jx = jx[y_pad:-y_pad, x_pad:-x_pad].real
    jy = jy[y_pad:-y_pad, x_pad:-x_pad].real

    return jx, jy
