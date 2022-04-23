import argparse

import numpy as np
import skimage
import skimage.io
from skimage import morphology
import alphashape
import scipy
from scipy import signal
from pykdtree.kdtree import KDTree


def clip_values(x, z, x_limit, z_limit):
    x_start = max(np.min(np.where(x > x_limit[0])[0]) - 1, 0)
    x_end = min(np.max(np.where(x < x_limit[1])[0]) + 1, len(x) - 1)

    x = x[x_start:x_end + 1]
    z = z[x_start:x_end + 1]

    z_start = max(np.min(np.where(z > z_limit[0])[0]) - 1, 0)
    z_end = min(np.max(np.where(z > z_limit[0])[0]) + 1, len(z) - 1)
    x = np.clip(x[z_start:z_end + 1], x_limit[0], x_limit[1])
    z = np.clip(z[z_start:z_end + 1], z_limit[0], z_limit[1] + 1)

    x = x - x_limit[0]
    z = z_limit[1] - z

    return x, z


def get_regionprops_3d(image, threshold, small_object_size=10):
    binary_3d = image > threshold
    binary_3d = skimage.morphology.remove_small_objects(binary_3d, small_object_size)

    label_3d = skimage.measure.label(binary_3d)
    regionprops_3d = skimage.measure.regionprops(label_3d)

    return regionprops_3d


def calculate_alpha_shape(regionprops_3d, mid_pt, centroid_filters, width, alpha):
    new_cen = []
    for prop in regionprops_3d:
        centroid = prop["centroid"]
        for filters in centroid_filters:
            if filters[0] == mid_pt:
                if filters[1] < centroid[0] < filters[2] and filters[3] < centroid[2] < filters[4]:
                    continue
        if mid_pt - width < centroid[1] < mid_pt + width:
            new_cen.append([centroid[2], centroid[0]])
    new_cen = np.array(new_cen)

    alpha_shape = alphashape.alphashape(new_cen, alpha=alpha)
    coords = np.array(alpha_shape.exterior.coords)

    return coords


def filter_coordinates(coords, shape, edge_width):
    filtered = []
    for coord in coords:
        if coord[1] > shape[0] - edge_width:
            continue
        elif coord[1] < edge_width:
            continue
        elif coord[0] < edge_width:
            continue
        elif coord[0] > shape[2] - edge_width:
            continue
        else:
            filtered.append(coord)
    return np.array(filtered)


def smooth_coordinates(filtered, start_point, total_vol):
    sort_idx = np.argsort(filtered[:, 0])
    smooth_pts_z = scipy.signal.savgol_filter(filtered[sort_idx, 1], 3, 1)
    smooth_pts_x = scipy.signal.savgol_filter(filtered[sort_idx, 0], 3, 1)

    reslice_x, reslice_z = clip_values(smooth_pts_x, smooth_pts_z,
                                       [start_point[2], start_point[2] + total_vol[2] - 1],
                                       [start_point[0], start_point[0] + total_vol[0] - 1])
    return reslice_x, reslice_z


def save_pts_list(pts_list, suffix=""):
    f = open(f"{suffix}pointcloud.txt", "a")
    f.write("3d=true\n")
    f.write("polyline=true\n")
    f.write("fit=false\n")
    f.write("\n")
    for pt_list in pts_list:
        np.savetxt(f, pt_list, fmt='%.5f', delimiter=" ")
        f.write("\n")
    f.close()


def generate_volume(pts_list, total_vol):
    y, x = np.mgrid[0:total_vol[1], 0:total_vol[2]]

    distance_list = []
    indices_list = []
    for i in range(len(pts_list)):
        pt_list = pts_list[i][:, [1, 2, 0]]
        data = np.c_[pt_list[:, 1].astype(np.float32).ravel(),
                     pt_list[:, 0].astype(np.float32).ravel()]
        tree = KDTree(data, leafsize=16)

        pts = np.array(np.c_[y.ravel(), x.ravel()], dtype=np.float32)

        distances, indices = tree.query(pts, k=1, distance_upper_bound=400.0)
        distance_list.append(distances)
        indices_list.append(indices)

        del data
        del tree

    z_threshold = np.zeros_like(distance_list[0])

    for i in range(z_threshold.shape[0]):
        out = 0
        distance = 0
        for j in range(len(distance_list)):
            if distance_list[j][i] == np.inf:
                continue
            if distance_list[j][i] == 0:
                out = distance_list[j][i]
                distance = 1
                break
            else:
                out += pts_list[j][indices_list[j][i]][0] * (1 / distance_list[j][i])
                distance += (1 / distance_list[j][i])
        z_threshold[i] = out / distance

    z_threshold = np.reshape(z_threshold, (1, total_vol[1], total_vol[2]))
    z, _, _ = np.mgrid[0:total_vol[0], 0:total_vol[1], 0:total_vol[2]]
    volume = z <= z_threshold

    return volume


def load_image(arg, axis=0):
    if type(arg) == list:
        images = []
        for file in arg:
            images.append(skimage.io.imread(file))
        return np.concatenate(images, axis=axis)
    else:
        return skimage.io.imread(arg)


def realign_image(image, z_reverse=False, z_start=None, z_end=None,
                  y_start=None, y_end=None, x_start=None, x_end=None):
    _z_start = z_start if z_start else 0
    _z_end = z_end if z_end else image.shape[0]

    _y_start = y_start if y_start else 0
    _y_end = y_end if y_end else image.shape[1]

    _x_start = x_start if x_start else 0
    _x_end = x_end if x_end else image.shape[2]

    _image = image[_z_start:_z_end, _y_start:_y_end, _x_start:_x_end]

    if z_reverse:
        _image = -image[::-1, ...]

    return _image


def imadjust(image, in_bound=(0.001, 0.999), out_bound=(0, 1)):
    """
    See https://stackoverflow.com/questions/39767612/what-is-the-equivalent-of-matlabs-imadjust-in-python/44529776#44529776
    image : input one-layer image (numpy array)
    in_bound  : src image bounds
    out_bound : dst image bounds
    output : output img
    """
    assert len(image.shape) == 2, 'Input image should be 2-dims'
    image_dtype = image.dtype

    if image_dtype == 'uint8':
        range_value = 255
    elif image_dtype == 'uint16':
        range_value = 65535
    else:
        range_value = np.max(image)

    # Compute in and out limits
    in_bound = np.percentile(image, np.multiply(in_bound, 100))
    out_bound = np.multiply(out_bound, range_value)

    # Stretching
    scale = (out_bound[1] - out_bound[0]) / (in_bound[1] - in_bound[0])

    image = image - in_bound[0]
    image[image < 0] = 0

    output = (image) * scale + out_bound[0]
    output[output > out_bound[1]] = out_bound[1]

    output = output.astype(image_dtype)

    return output


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_image",
                        help="File path or list of file paths to generate the 3D volume from. If a list is given, the images are concatenates along the 0-axis.",
                        nargs='+')
    parser.add_argument("--start-point",
                        help="Starting point from which to generate the volume from. Default: (0, 0, 0)",
                        default=(0, 0, 0),
                        type=int,
                        nargs='+')
    parser.add_argument("--total-vol",
                        help="Total volume of the region of interest. Default: shape of image",
                        type=int,
                        nargs='+')
    parser.add_argument("--realign",
                        help="Tuple or list containing value to determine if there is a need to reverse or crop the image. Default: None",
                        default=None,
                        type=int,
                        nargs='+')
    parser.add_argument("-t", "--threshold",
                        help="Binarization threshold value. Default: 750",
                        type=float, default=750)
    parser.add_argument("--small-object-size",
                        help="Threshold to filter out small objects. Default: 10",
                        type=float, default=10)
    parser.add_argument("--width",
                        help="Width of volume to calculate the profile from. Default: 100",
                        type=int, default=100)
    parser.add_argument("--n_volume",
                        help="Number of volumes to calculate the profile from. Default: 3",
                        type=int, default=3)
    parser.add_argument("--alpha",
                        help="Alpha used in the determination of the alpha shape. Default: 0.02",
                        type=float, default=0.02)
    parser.add_argument("--centroid_filters",
                        help="Tuple or list containing value to filter the centroids based on positions. Default: None",
                        default=None,
                        nargs='+')
    parser.add_argument("--edge-width",
                        help="Edge coordinates to remove and only leave coordinates that define the surface curvature. Default: 20",
                        type=int, default=20)

    return parser.parse_args()


def generate_3d_model():
    # TODO: this needs to be refactored as there are too many inputs
    args = parse_args()

    if args.centroid_filters is None:
        centroid_filters = ()
    else:
        centroid_filters = args.centroid_filters

    image = load_image(args.input_image)
    if args.realign:
        image = realign_image(image, z_reverse=args.realign[0],
                              z_start=args.realign[1], z_end=args.realign[2],
                              y_start=args.realign[3], y_end=args.realign[4],
                              x_start=args.realign[5], x_end=args.realign[6])
    shape = image.shape

    start_point = args.start_point
    if args.total_vol is None:
        total_vol = shape
    else:
        total_vol = args.total_vol

    regionprops_3d = get_regionprops_3d(image, args.threshold, small_object_size=args.small_object_size)

    pts_list = []
    mid_pts = np.rint(np.linspace(start_point[1], start_point[1] + total_vol[1] - 1, args.n_volume, endpoint=True)).astype(
        np.int32)

    for mid_pt in mid_pts:
        coords = calculate_alpha_shape(regionprops_3d, mid_pt, centroid_filters, args.width, args.alpha)
        filtered = filter_coordinates(coords, shape, args.edge_width)

        reslice_x, reslice_z = smooth_coordinates(filtered, start_point, total_vol)

        out_list = np.stack((reslice_z, reslice_x, np.repeat(mid_pt - start_point[1], len(reslice_x))))
        out_list = out_list.T
        pts_list.append(out_list)

    save_pts_list(pts_list, suffix='')
    vol = generate_volume(pts_list, total_vol)

    np.savez(f"volume.npz", vol)

# if __name__ == '__main__':
#     pass
#     # generate_3d_model()
