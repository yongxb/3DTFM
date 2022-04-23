import os

import numpy as np
from regularization import l_curve, tikhonov
import matplotlib.pyplot as plt
import scipy.linalg

import pandas as pd
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_folder", help="Parent folder containing all the needed folder/files")
    parser.add_argument("--displacement",
                        help=".txt file containing the comma separated interpolated displacement values from ANSYS. Default: interpolated_displacement.txt",
                        default="interpolated_displacement.txt")
    parser.add_argument("--subfolder",
                        help="Subfolder inside <output> that is used to store the results. Default: pkl",
                        default="pkl")
    parser.add_argument("--input-csv",
                        help=".csv file containing the nodal information. Default: <folder_basename>_top_nodes.csv")
    parser.add_argument("--svd-file",
                        help=".npz file containing the output of the singular value decomposition. Will be created if not found in parent folder. Default: svd.npz",
                        default="svd.npz")
    parser.add_argument("--nodal-force",
                        help="Nodal force used in FEM model. Default: 1e-8",
                        default="1e-8")
    parser.add_argument("--pixel-um",
                        help="Pixel to µm factor. Default: 0.275",
                        type=float, default=0.275)
    parser.add_argument("--save-plot",
                        help="File name to save l curve plot. If set to None, the plot is not generated. Default: l_curve.png",
                        default="l_curve.png")
    parser.add_argument("--save-forces",
                        help="File name to save force solution. Default: forces.csv",
                        default="forces.csv")

    return parser.parse_args()


def plot_lcurve(rho, eta, reg_param, reg_corner, save_location, num_elems=10):
    plt.figure(figsize=(8, 6), dpi=300)
    plt.loglog(rho, eta)

    idx = np.round(np.linspace(0, len(rho) - 1, num_elems)).astype(int)
    plt.scatter(rho[idx], eta[idx], color='orange', marker='x')
    ax = plt.gca()

    for i, (reg2, eta2, rho2) in enumerate(zip(reg_param[idx], eta[idx], rho[idx])):
        ax.annotate("{:.2e}".format(reg2[0]), (rho2, eta2))

    plt.xlabel("Residual norm")
    plt.ylabel("Solution norm")
    plt.title("L-curve, Tikh.corner at {:.4e}".format(reg_corner[0]))
    plt.savefig(save_location)


def save_force_solution(corner, u, s, v, b, node_ids, output_filepath, factor=1):
    x_lambda, _, _ = tikhonov(u, s, v.T, b, corner)

    x_lambda = x_lambda * factor
    x_lambda = np.reshape(x_lambda, (3, -1))

    output = np.hstack((node_ids[:, np.newaxis], x_lambda.T))

    np.savetxt(output_filepath, output, delimiter=',')


def calculate_force_solution():
    args = parse_args()

    basename = os.path.split(args.input_folder)[1]
    solution_path = args.input_folder

    if args.input_csv:
        input_csv = args.input_csv
    else:
        input_csv = f'{basename}_top_nodes.csv'

    factor = eval(args.nodal_force) * eval(args.pixel_um) * 1e-6

    top_node_dataset = pd.read_csv(os.path.join(solution_path, input_csv), header=None)
    top_node_dataset = top_node_dataset.rename(columns={0: "NodeID", 1: "X", 2: "Y", 3: "Z"})
    top_node_ids = top_node_dataset.to_numpy()[:, 0]

    displacement_data_path = os.path.join(solution_path, args.displacement)
    disp_data_set = pd.read_table(displacement_data_path)
    disp_data_set = np.array(disp_data_set.iloc[0:, 1:-1])
    b = np.reshape(disp_data_set.T, (-1, 1))

    if os.path.isfile(os.path.join(solution_path, args.svd_file)):
        print("SVD file has been specified at {}".format(os.path.join(solution_path, args.svd_file)))
        npzfile = np.load(os.path.join(solution_path, args.svd_file))
        u = npzfile['u']
        s = npzfile['s']
        v = npzfile['v']
    else:
        num_files = len(top_node_ids)

        # initialize A matrix of inverse problem by columnwise combination of each file data, reordering as x, y, then z components aross the columns of A
        # i.e.,        [DxFx.. DxFy.. DxFz..]
        #              [DyFx.. DyFy.. DyFz..]
        #              [DzFx.. DzFy.. DzFz..]
        # each column represents a node: Ds are displacements and Fs are the force components applied to the face node to derive the column
        for k, nodeId in enumerate(top_node_ids):
            file_path = os.path.join(solution_path, args.subfolder, f"solutionSet_{int(nodeId):06d}.pkl")
            with open(file_path, "rb") as file:
                displacement_dict = pickle.load(file)

            if k == 0:
                A = np.empty((displacement_dict[0].shape[0] * 3, num_files * 3), dtype=float)

            for t in range(3):
                displacement = np.reshape(displacement_dict[t].T, (-1,))
                index = t * num_files + k
                A[:, index] = displacement

        u, s, v = scipy.linalg.svd(A, full_matrices=False, compute_uv=True, check_finite=True, lapack_driver='gesvd')
        np.savez(os.path.join(solution_path, args.svd_file), u=u, s=s, v=v)

        del A

    reg_corner, rho, eta, reg_param = l_curve(u, s, b)

    if args.save_plot.lower() != "none":
        plot_lcurve(rho, eta, reg_param, reg_corner, os.path.join(solution_path, args.save_plot))

    save_force_solution(reg_corner, u, s, v, b, top_node_ids, os.path.join(solution_path, args.save_forces), factor=factor)


if __name__ == '__main__':
    calculate_force_solution()
