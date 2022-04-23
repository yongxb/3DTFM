from mayavi import mlab
import numpy as np
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", help=".csv file containing the forces")
    parser.add_argument("-v", "--visualization",
                        help="Type of visualization. Accepted values are [forces, orthogonal, flatten_orthogonal, flatten_z, flatten_x]",
                        default="forces")
    parser.add_argument("--flatten",
                        help="Whether to show the forces in its 3D space or as a flattened surface. Default: False  ",
                        default="False")
    parser.add_argument("--z-pos",
                        help="Whether to plot z position on a wireframe grid. Only valid if --flatten is True. Default: False",
                        default="False")
    return parser.parse_args()


def visualization():
    # accepted_visualization = ["forces", "orthogonal", "flatten_orthogonal", "flatten_z", "flatten_x"]
    args = parse_args()

    dataset = pd.read_csv(args.input_file)

    zero_array = np.zeros_like(dataset["Z"])

    parameters = []
    fig = mlab.figure(bgcolor=(0.15, 0.15, 0.15), size=(1920, 1080))


    if args.flatten.lower() == "true":
        is_flatten = True
        parameters = parameters + [dataset["posX"], dataset["posY"], zero_array]

        if args.z_pos.lower() == "true":
            y, x = np.mgrid[np.min(dataset["posY"]):np.max(dataset["posY"]) + 1,
                   np.min(dataset["posX"]):np.max(dataset["posX"]) + 1]
            surf = np.zeros_like(y)
            ymin = int(np.min(dataset["posY"]))
            xmin = int(np.min(dataset["posX"]))
            for ix, iy in zip(x.ravel(), y.ravel()):
                idx = np.nonzero(np.logical_and(dataset["posX"] == (ix), dataset["posY"] == (iy)).to_numpy())[0][0]
                surf[int(iy - ymin), int(ix - xmin)] = dataset.iloc[idx, :]["Z"]

            s = mlab.surf(surf.T, representation="wireframe", warp_scale=0, figure=fig)
            s.module_manager.scalar_lut_manager.lut_mode = 'coolwarm'
            fig.children[0].origin = [xmin, ymin, 0.]
            fig.scene.z_plus_view()
            fig.scene.camera.compute_view_plane_normal()

    elif args.flatten.lower() == "false":
        is_flatten = False
        parameters = parameters + [dataset["X"], dataset["Y"], dataset["Z"]]
    else:
        print("Unknown value for --flatten. Setting to False")
        is_flatten = False
        parameters = parameters + [dataset["X"], dataset["Y"], dataset["Z"]]

    if args.visualization.lower() == "forces":
        if is_flatten:
            print("--flatten is not supported for this visualization. Exiting")
            return
        force = [dataset["ForceX"], dataset["ForceY"], dataset["ForceZ"]]
        mlab.quiver3d(*(parameters+force), mode='arrow')

    elif args.visualization.lower() == "orthogonal" and not is_flatten:
        force = [dataset["ForceNormX"] * dataset["NormX.X"],
                 dataset["ForceNormX"] * dataset["NormX.Y"],
                 dataset["ForceNormX"] * dataset["NormX.Z"]]
        mlab.quiver3d(*(parameters+force), mode='arrow')

        force = [dataset["ForceNormY"] * dataset["NormY.X"],
                 dataset["ForceNormY"] * dataset["NormY.Y"],
                 dataset["ForceNormY"] * dataset["NormY.Z"]]
        mlab.quiver3d(*(parameters+force), mode='arrow')

        force = [dataset["ForceNormZ"] * dataset["NormZ.X"],
                 dataset["ForceNormZ"] * dataset["NormZ.Y"],
                 dataset["ForceNormZ"] * dataset["NormZ.Z"]]
        mlab.quiver3d(*(parameters+force), mode='arrow')

    elif args.visualization.lower() == "orthogonal" and is_flatten:
        force = [dataset["ForceNormX"], zero_array, zero_array]
        mlab.quiver3d(*(parameters+force), mode='arrow', figure=fig)
        force = [zero_array, dataset["ForceNormY"], zero_array]
        mlab.quiver3d(*(parameters+force), mode='arrow', figure=fig)
        force = [zero_array, zero_array, dataset["ForceNormZ"]]
        mlab.quiver3d(*(parameters+force), mode='arrow', figure=fig)

    elif args.visualization.lower() == "orthogonal_x":
        if is_flatten:
            force = [dataset["ForceNormX"], zero_array, zero_array]
        else:
            force = [dataset["ForceNormX"] * dataset["NormX.X"],
                     dataset["ForceNormX"] * dataset["NormX.Y"],
                     dataset["ForceNormX"] * dataset["NormX.Z"]]
        q = mlab.quiver3d(*(parameters + force), mode='arrow', figure=fig)
        range_value = np.max((np.abs(np.min(dataset["ForceNormX"])), np.abs(np.max(dataset["ForceNormX"]))))
        q.glyph.color_mode = 'color_by_scalar'
        q.mlab_source.dataset.point_data.scalars = dataset["ForceNormX"].to_numpy()
        q.module_manager.scalar_lut_manager.use_default_range = False
        q.module_manager.scalar_lut_manager.data_range = [-range_value, range_value]
        q.glyph.glyph_source.glyph_source = q.glyph.glyph_source.glyph_dict['cone_source']

    elif args.visualization.lower() == "orthogonal_y":
        if is_flatten:
            force = [zero_array, dataset["ForceNormY"], zero_array]
        else:
            force = [dataset["ForceNormY"] * dataset["NormY.X"],
                     dataset["ForceNormY"] * dataset["NormY.Y"],
                     dataset["ForceNormY"] * dataset["NormY.Z"]]
        q = mlab.quiver3d(*(parameters+force), mode='arrow', figure=fig)
        range_value = np.max((np.abs(np.min(dataset["ForceNormY"])), np.abs(np.max(dataset["ForceNormY"]))))
        q.glyph.color_mode = 'color_by_scalar'
        q.mlab_source.dataset.point_data.scalars = dataset["ForceNormY"].to_numpy()
        q.module_manager.scalar_lut_manager.use_default_range = False
        q.module_manager.scalar_lut_manager.data_range = [-range_value, range_value]
        q.glyph.glyph_source.glyph_source = q.glyph.glyph_source.glyph_dict['cone_source']

    elif args.visualization.lower() == "orthogonal_z":
        if is_flatten:
            force = [zero_array, zero_array, dataset["ForceNormZ"]]
        else:
            force = [dataset["ForceNormZ"] * dataset["NormZ.X"],
                     dataset["ForceNormZ"] * dataset["NormZ.Y"],
                     dataset["ForceNormZ"] * dataset["NormZ.Z"]]
        q = mlab.quiver3d(*(parameters+force), mode='arrow', figure=fig)
        range_value = np.max((np.abs(np.min(dataset["ForceNormZ"])), np.abs(np.max(dataset["ForceNormZ"]))))
        q.glyph.color_mode = 'color_by_scalar'
        q.mlab_source.dataset.point_data.scalars = dataset["ForceNormZ"].to_numpy()
        q.module_manager.scalar_lut_manager.use_default_range = False
        q.module_manager.scalar_lut_manager.data_range = [-range_value, range_value]
        q.glyph.glyph_source.glyph_source = q.glyph.glyph_source.glyph_dict['cone_source']
    else:
        print(f"{args.visualization} not supported. Exiting.")
        return

    mlab.show()


if __name__ == '__main__':
    visualization()
