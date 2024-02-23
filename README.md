# current-reconstruction

This repo implements a Fourier transform inversion of the 2D Biot-Savart law to reconstruct a 2D current density distribution from an image of the out-of-plane magnet field B_z, or a magnetic flux signal given by B_z convolved with a magnetic
sensor's  spread function or imaging kernel.

The method is based on "Using a magnetometer to image a two‐dimensional current distribution",
J. Appl. Phys. 65, 361–372 (1989) https://doi.org/10.1063/1.342549. A free PDF of the paper is available at
https://www.vanderbilt.edu/lsp/documents/jap-roth-using-89.pdf.

A sample dataset from a scanning SQUID current imaging experiment is provided in `./sample_data/`. The dataset includes the point spread function of the SQUID magnetometer (`psf.npz`) obtained by imaging a small magnetic dipole oriented out of the plane. The current imaging data (`mag.npz`) was obtained by sourcing a ~600 Hz 1 µA AC current through a small device and measuring the AC flux picked up by the SQUID using a lockin amplifier. The reconstructed current density is in arbitrary units because the magnetic moment of the dipole used to calibrate the SQUID point spread function is unknown.

![Image](./sample_data/current_reconstruction.png)