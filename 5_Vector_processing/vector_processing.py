import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import math
import cupy as cp
import cupyx
from cupyx.scipy import ndimage


def load_volume(file):
    file_vol = np.load(file)
    vol = file_vol['arr_0']
    return vol


def load_error(file):
    file_error = np.load(file)
    error = file_error['error']
    return error


def load_vectors(file):
    npzfile = np.load(file)
    
    vx = npzfile['vx']
    vy = npzfile['vy']
    vz = npzfile['vz']
    del npzfile
    
    return vx, vy, vz


def subtract_median(vectors, z_start, z_end, edge=50):
    median = np.nanmedian(vectors[z_start:z_end, edge:-edge, edge:-edge])
    vectors = vectors - median

    return vectors


def subtract_z_movement(vz, peak_pos, width, z_end_pt, output_file=None):
    vzline = vz[:z_end_pt, :, peak_pos:peak_pos + width]
    z, _, _ = np.indices(vzline.shape)
    poly_val = np.polyfit(z.ravel(), vzline.ravel(), 1)
    poly_fn = np.poly1d(poly_val)

    if output_file:
        plt.figure(figsize=(20, 15))
        plt.plot(z.ravel(), vzline.ravel(), '.')
        plt.plot(np.arange(z_end_pt), poly_fn(np.arange(z_end_pt)))
        plt.savefig(output_file)

    z, _, _ = np.indices(vz.shape)
    z_correction = poly_fn(z)
    vz = vz - z_correction
    del z, z_correction

    return vz


def filter_by_volume(vol, vx, vy, vz):
    vx[vol == 0] = np.nan
    vy[vol == 0] = np.nan
    vz[vol == 0] = np.nan
    return vx, vy, vz


def filter_by_error(error, threshold, vx, vy, vz):
    vx[error > threshold] = np.nan
    vy[error > threshold] = np.nan
    vz[error > threshold] = np.nan
    return vx, vy, vz


def gaussian_kernel_1d(sigma, radius=None):
    if radius is None:
        radius = math.ceil(2 * sigma)

    output_kernel = np.mgrid[-radius:radius + 1]
    output_kernel = np.exp((-(1 / 2) * (output_kernel ** 2)) / (sigma ** 2))
    output_kernel = output_kernel / np.sum(output_kernel)

    return output_kernel


def downsample(image, sigma=1, skip=4):
    image_gpu = cp.asarray(image)
    kernel = cp.asarray(gaussian_kernel_1d(sigma), dtype=cp.float32)
    radius = math.ceil(2 * sigma)

    # gaussian smoothing
    image_gpu = cupyx.scipy.ndimage.convolve(image_gpu, cp.reshape(kernel, (2 * radius + 1, 1, 1)), mode='reflect')
    image_gpu = cupyx.scipy.ndimage.convolve(image_gpu, cp.reshape(kernel, (1, 2 * radius + 1, 1)), mode='reflect')
    image_gpu = cupyx.scipy.ndimage.convolve(image_gpu, cp.reshape(kernel, (1, 1, 2 * radius + 1)), mode='reflect')

    image_cpu = image_gpu.get()
    del image_gpu

    return image_cpu[::skip, ::skip, ::skip]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_folder", help="Parent folder containing all the needed folder/files")
    parser.add_argument("--vector-file",
                        help=".npz file containing the vectors obtained from 3D optical flow. Default: vectors.npz",
                        default="vectors.npz")
    parser.add_argument("--error-file",
                        help=".npz file containing the errors obtained from 3D optical flow. Default: None",
                        default=None)
    parser.add_argument("--vol-file",
                        help=".npz file containing the volume mask. Default: None",
                        default=None)
    parser.add_argument("--z-volume-start",
                        help="Specifies the z start position to define the volume to calculate the median values from. Default: 25",
                        type=int, default=25)
    parser.add_argument("--z-volume-end",
                        help="Specifies the z end position to define the volume to calculate the median values from. Default: 50",
                        type=int, default=50)
    parser.add_argument("--edge-width",
                        help="Specifies the distance from the edge to ignore during median value calculation. Default: 50",
                        type=int, default=50)
    parser.add_argument("--z-correction",
                        help="Specifies the z correction method. Supported values are [median, profile]. Default: median",
                        default="median")
    parser.add_argument("--profile-pos",
                        help="Specifies the x position of the line with which to obtain the profile. Default: None",
                        type=int, default=None)
    parser.add_argument("--profile-width",
                        help="Specifies the width of the line profile in the x direction. Default: None",
                        type=int, default=None)
    parser.add_argument("--profile-z-end",
                        help="Specifies the height of the profile from the base of the structure. Default: None",
                        type=int, default=None)
    parser.add_argument("--sigma",
                        help="Sigma of Gaussian kernel used during downsampling. Default: 1",
                        type=float, default=1)
    parser.add_argument("--step",
                        help="Downsampling step size. Default: 4",
                        type=int, default=4)
    parser.add_argument("--output-file",
                        help="Output .txt file containing forces that can be imported into ANSYS. Default: ansys_displacement.txt",
                        default="ansys_displacement.txt")

    return parser.parse_args()


def filter_vectors():
    args = parse_args()

    vx, vy, vz = load_vectors(os.path.join(args.input_folder, args.vector_file))

    vx = subtract_median(vx, args.z_volume_start, args.z_volume_end, edge=args.edge_width)
    vy = subtract_median(vy, args.z_volume_start, args.z_volume_end, edge=args.edge_width)

    if args.z_correction.lower() == "profile":
        assert args.profile_pos is not None
        assert args.profile_width is not None
        assert args.profile_z_end is not None
        vz = subtract_z_movement(vz, args.profile_pos, args.profile_width, args.profile_z_end, output_file=None)
    elif args.z_correction.lower() == "median":
        vz = subtract_median(vz, args.z_volume_start, args.z_volume_end, edge=args.edge_width)

    vz = np.float32(vz)
    vy = np.float32(vy)
    vx = np.float32(vx)

    if args.vol_file is not None:
        vol = load_volume(args.vol_file)
        vx, vy, vz = filter_by_volume(vol, vx, vy, vz)

    if args.error_file is not None:
        if args.threshold is None:
            threshold = 1
        else:
            threshold = args.threshold
        error = load_error(args.error_file)
        vx, vy, vz = filter_by_error(error, threshold, vx, vy, vz)

    shape = vx.shape
    vx = downsample(vx, sigma=args.sigma, skip=args.skip)
    vy = downsample(vy, sigma=args.sigma, skip=args.skip)
    vz = downsample(vz, sigma=args.sigma, skip=args.skip)

    z, y, x = np.mgrid[0:shape[0]:args.skip, 0:shape[1]:args.skip, 0:shape[2]:args.skip]
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    mask = np.logical_or(np.logical_or(np.isnan(vx), np.isnan(vy)), np.isnan(vz))
    out = np.stack((vx[~mask], vy[~mask], vz[~mask],
                    x[~mask], y[~mask], z[~mask])).T

    f = open(args.output_file, "w")
    np.savetxt(f, out, fmt=['%.5f'] * 3 + ['%d'] * 3, delimiter=", ")
    f.close()


if __name__ == '__main__':
    filter_vectors()

