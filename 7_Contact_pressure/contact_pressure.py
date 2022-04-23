import os

import numpy as np
import pandas as pd
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_folder", help="Parent folder containing all the needed folder/files")
    parser.add_argument("--element-filename",
                        help=".json file containing the elements in the ANSYS model. Default: elements.json",
                        default="elements.json")
    parser.add_argument("--nodes-csv",
                        help=".csv file containing the nodal information for all nodes. Default: <folder_basename>_nodes.csv")
    parser.add_argument("--top-nodes-csv",
                        help=".csv file containing the nodal information for the top surface nodes. Default: <folder_basename>_top_nodes.csv")
    parser.add_argument("--forces",
                        help=".csv file containing the solved forces. Default: forces.csv", default="forces.csv")
    parser.add_argument("--output-csv",
                        help="Output .csv file containing the contact pressure at each node. Default: contact_pressure.csv",
                        default="contact_pressure.csv")
    parser.add_argument("--width", help="Specifies the width between the nodes. Default: 0.02",
                        type=float, default=0.02)
    parser.add_argument("--pixel-um", help="Specifies the pixel to µm conversion. Default: 0.275",
                        type=float, default=0.275)

    return parser.parse_args()


def calculate_pressure(elements, node_ids, force_ids, points_array, force_array, pixel_to_um):
    node_pressure = []
    node_normal = []
    node_x = []
    node_y = []
    for ele_id, ele_list in elements.items():
        hull_areas = []
        hull_cross = []
        hull_x = []
        hull_y = []
        pressure = []

        for ele in ele_list:
            indices = np.in1d(node_ids, ele).nonzero()[0]
            points = points_array[indices, 1:4].astype(np.float32)

            # calculate area by triangulation of points
            if len(indices) == 3:
                cross_pdt = np.cross(points[0] - points[1], points[2] - points[1])
                area = 0.5 * np.linalg.norm(cross_pdt)
                if np.sign(cross_pdt[2]) == -1:
                    cross_pdt = -cross_pdt
            elif len(indices) == 4:
                area = 0
                cross_pdt = 0
                start_point = 0
                dist = np.linalg.norm(points - points[0], axis=1)
                dist_argmax = np.argmax(dist)

                # verify if pair of points are the furthest apart
                dist_check = np.linalg.norm(points - points[dist_argmax], axis=1)
                if np.argmax(dist_check) != 0:
                    start_point = dist_argmax
                    dist_argmax = np.argmax(dist_check)

                for i in range(len(indices)):
                    if i == dist_argmax or i == start_point:
                        continue
                    else:
                        cross = np.cross(points[i] - points[start_point],
                                         points[dist_argmax] - points[start_point])
                        area += 0.5 * np.linalg.norm(cross)
                        if np.sign(cross[2]) == -1:
                            cross_pdt += -cross
                        else:
                            cross_pdt += cross

                        temp = points[i] - points[start_point]
                        if np.argmin(np.abs(temp[:2])) == 0:
                            if np.sign(temp[1]) == -1:
                                hull_y.append(-temp)
                            else:
                                hull_y.append(temp)
                        elif np.argmin(np.abs(temp[:2])) == 1:
                            if np.sign(temp[0]) == -1:
                                hull_x.append(-temp)
                            else:
                                hull_x.append(temp)
                cross_pdt = cross_pdt / 2

            area = area * pixel_to_um * pixel_to_um * 1e6 # (1e3)^2 to correct as Ansys treats 1000px as 1 unit.
            print(area)
            hull_areas.append(area)
            hull_cross.append(cross_pdt)

            force_indices = np.in1d(force_ids, ele).nonzero()[0]
            forces = force_array[force_indices, 1:]
            pressure.append(np.mean(forces, axis=0) / area)

        node_pressure.append(np.mean(pressure, axis=0))
        mean_hull_cross = np.mean(hull_cross, axis=0)
        mean_hull_x = np.mean(hull_x, axis=0)
        mean_hull_y = np.mean(hull_y, axis=0)
        node_normal.append(mean_hull_cross / np.linalg.norm(mean_hull_cross))

        # perfrom Graham-Schmidt orthonormalization
        cross_x = mean_hull_x - np.dot(mean_hull_cross, mean_hull_x) / np.dot(mean_hull_cross, mean_hull_cross) * mean_hull_cross
        node_x.append(cross_x / np.linalg.norm(cross_x))
        cross_y = mean_hull_y - \
                  np.dot(mean_hull_cross, mean_hull_y) / np.dot(mean_hull_cross, mean_hull_cross) * mean_hull_cross - \
                  np.dot(cross_x, mean_hull_y) / np.dot(cross_x, cross_x) * cross_x
        node_y.append(cross_y / np.linalg.norm(cross_y))

    return node_x, node_y, node_pressure, node_normal


def get_positions(nodes, width):
    round_y = width * np.round((nodes["Y"] - np.min(nodes["Y"])) / width) + np.round(np.min(nodes["Y"]), 3)
    unique_y = np.unique(round_y)
    num_rows = len(unique_y)
    num_cols = int(len(nodes["Y"]) / num_rows)

    x_pos = np.zeros_like(round_y)
    y_pos = np.zeros_like(round_y)

    for i in range(num_rows):
        idx = np.nonzero(round_y.to_numpy() == unique_y[i])[0]
        y_pos[idx] = i

        x_values = nodes["X"].to_numpy()[idx]
        x_idx = np.argsort(x_values)
        x_pos[idx[x_idx]] = np.arange(num_cols)

    return x_pos, y_pos


def contact_pressure():
    args = parse_args()

    basename = os.path.split(args.input_folder)[1]
    solution_path = args.input_folder

    if args.nodes_csv:
        nodes_file_name = args.nodes_csv
    else:
        nodes_file_name = f"{basename}_nodes.csv"

    if args.top_nodes_csv:
        top_nodes_file_name = args.top_nodes_csv
    else:
        top_nodes_file_name = f"{basename}_top_nodes.csv"

    whole_face_dataset = pd.read_csv(os.path.join(solution_path, nodes_file_name), header=None)
    whole_face_array = whole_face_dataset.to_numpy()
    node_ids = whole_face_array[:, 0]
    force_dataset = pd.read_csv(os.path.join(solution_path, args.forces), header=None)
    force_array = force_dataset.to_numpy()
    force_ids = force_array[:, 0]

    face_nodes_dataset = pd.read_csv(os.path.join(solution_path, top_nodes_file_name), header=None)
    face_nodes_dataset = face_nodes_dataset.rename(columns={0: "NodeID", 1: "X", 2: "Y", 3: "Z"})

    x_pos, y_pos = get_positions(face_nodes_dataset, args.width)

    pos_dataset = pd.DataFrame({
        "posX": x_pos,
        "posY": y_pos, })
    face_nodes_dataset = pd.concat([face_nodes_dataset, pos_dataset], axis=1)

    # %%
    with open(os.path.join(solution_path, args.element_filename), "r") as file:
        elements = json.load(file)

    node_x, node_y, node_pressure, node_normal = calculate_pressure(elements, node_ids, force_ids, whole_face_array,
                                                                    force_array, args.pixel_um)
    node_pressure = np.array(node_pressure)
    node_normal = np.array(node_normal)
    node_x = np.array(node_x)
    node_y = np.array(node_y)

    force_dataset = pd.DataFrame({
        "ForceX": node_pressure[:, 0],
        "ForceY": node_pressure[:, 1],
        "ForceZ": node_pressure[:, 2],
        "NormX.X": node_x[:, 0],
        "NormX.Y": node_x[:, 1],
        "NormX.Z": node_x[:, 2],
        "NormY.X": node_y[:, 0],
        "NormY.Y": node_y[:, 1],
        "NormY.Z": node_y[:, 2],
        "NormZ.X": node_normal[:, 0],
        "NormZ.Y": node_normal[:, 1],
        "NormZ.Z": node_normal[:, 2],
        "ForceNormX": [np.dot(x, y) for x, y in zip(node_pressure, node_x)],
        "ForceNormY": [np.dot(x, y) for x, y in zip(node_pressure, node_y)],
        "ForceNormZ": [np.dot(x, y) for x, y in zip(node_pressure, node_normal)],
    })

    face_nodes_dataset = pd.concat([face_nodes_dataset, force_dataset], axis=1)
    face_nodes_dataset.to_csv(os.path.join(solution_path, args.output_csv), index=False)
    print(f"Contact pressure save to: {os.path.join(solution_path, args.output_csv)}")


if __name__ == '__main__':
    contact_pressure()
