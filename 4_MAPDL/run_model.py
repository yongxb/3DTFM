import os
import re
import pandas as pd
import subprocess
import shlex
import shutil
import pickle
from ansys.mapdl import reader as pymapdl_reader

from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_folder", help="Folder containing the ANSYS .dat model")
    parser.add_argument("--input-csv", help="CSV file containing the nodal information. Default: <folder_basename>_top_nodes.csv")
    parser.add_argument("--input-filename", help="Name of input ANSYS .dat model. Default: ds.dat", default="ds.dat")
    parser.add_argument("--output-filename", help="Name of temporary ANSYS .dat model. Default: ds_temp.dat", default="ds_temp.dat")
    parser.add_argument("--output", help="Output folder to store the results and temporary files. Default: folder")
    parser.add_argument("--subfolder", help="Subfolder inside <output> that is used to store the results. Default: pkl", default="pkl")
    parser.add_argument("--points", help="Number of points in each batch. Default: 100", type=int, default=100)
    parser.add_argument("--nproc", help="Number of processors to use in Ansys Mechanical APDL. Default: 4", type=int, default=4)
    parser.add_argument("--mapdl-location", help="Location of MAPDL.exe. Default: \"C:\\Program Files\\ANSYS Inc\\ANSYS Student\\v211\\ansys\\bin\\winx64\\MAPDL.exe\"")

    return parser.parse_args()


def load_nodes(filepath):
    node_id_csv = pd.read_csv(filepath, header=None)
    node_ids = node_id_csv.to_numpy()
    nodes = node_ids[:, 0]

    return nodes


def load_apdl(filepath):
    with open(filepath, "r") as file:
        apdl_dat = file.readlines()
    return apdl_dat


def rebuild_apdl(apdl_dat, nodes, current_index, n_points):
    time_points = 1 + 3 * n_points

    for i, x in enumerate(apdl_dat):
        if x == '/wb,contact,end            !  done creating contacts\n':
            new_apdl = apdl_dat[:i + 1]
            for n in range(n_points):
                if n == 0:
                    new_apdl.extend([
                        "CMBLOCK,SELECTION,NODE,        1\n",
                        "(8i10)\n",
                        "{:>10}\n".format(int(nodes[current_index + n])),
                    ])
                else:
                    new_apdl.extend([
                        f"CMBLOCK,SELECTION_{n + 1},NODE,        1\n",
                        "(8i10)\n",
                        "{:>10}\n".format(int(nodes[current_index + n])),
                    ])
        if x == '/golist\n':
            i_temp = i
        if x == '/com,*********** Create Direct FE , Forces ****\n':
            new_apdl.extend(apdl_dat[i_temp:i])
            vari_num = int(re.findall(r'_CM\d+', apdl_dat[i + 1])[0][3:])

            for n in range(n_points):
                new_apdl.extend(
                    ["/com,*********** Create Direct FE , Forces ****\n",
                     f"CMBLOCK,_CM{vari_num + 2 * n}F,NODE,        1\n",
                     "(8i10)\n",
                     "{:>10}\n".format(int(nodes[current_index + n])),
                     ])

                for k, direction in enumerate(['x', 'y', 'z'], start=1):
                    new_apdl.extend([
                        '\n',
                        f"*DIM,_loadvari{vari_num + 2 * n}{direction},TABLE,{time_points},1,1,TIME,\n",
                        "! Time values\n"])
                    for t in range(time_points):
                        new_apdl.append(f"_loadvari{vari_num + 2 * n}{direction}({t + 1},0,1) = {t}.\n")

                    new_apdl.append("! Load values\n")
                    for t in range(time_points):
                        new_apdl.append(
                            f"_loadvari{vari_num + 2 * n}{direction}({t + 1},1,1) = {'1.e-008' if t == (k + 3 * n) else '0.'}\n")
                    new_apdl.append('\n')

        if x == '/gst,on,on\n':
            new_apdl.extend(apdl_dat[i:i + 32])

            for t in range(1, time_points + 1):
                if t == 1:
                    new_apdl.extend(
                        ["/com,****************************************************\n",
                         f"/com,******************* SOLVE FOR LS {t} OF {time_points} ****************\n", ])

                    for n in range(n_points):
                        new_apdl.extend([
                            "/com,**** Set Direct FE , Forces ****\n"
                            f"f,_CM{vari_num + 2 * n}F,fx,%_loadvari{vari_num + 2 * n}x%\n"
                            f"f,_CM{vari_num + 2 * n}F,fy,%_loadvari{vari_num + 2 * n}y%\n"
                            f"f,_CM{vari_num + 2 * n}F,fz,%_loadvari{vari_num + 2 * n}z%\n"
                        ])

                    new_apdl.extend([
                        "/nopr\n",
                        "/gopr\n",
                        "/nolist\n",
                        "nsub,1,1,1\n",
                        f"time,{t}.\n",
                        "outres,erase\n"
                        "outres,all,none\n"
                        "outres,nsol,all,\n"
                        "outres,rsol,all\n"
                        "outres,etmp,all\n"
                        "! *********** WB SOLVE COMMAND ***********\n"
                        "! check interactive state\n"
                        "*get,ANSINTER_,active,,int\n"
                        "*if,ANSINTER_,ne,0,then\n"
                        "/eof\n"
                        "*endif\n"
                        "solve\n"
                        "/com *************** Write FE CONNECTORS ********* \n"
                        "CEWRITE,file,ce,,INTE\n"
                        "/com,****************************************************\n"
                        f"/com,*************** FINISHED SOLVE FOR LS {t} *************\n"]
                    )
                else:
                    new_apdl.extend(
                        ["/com,****************************************************\n",
                         f"/com,******************* SOLVE FOR LS {t} OF {time_points} ****************\n",
                         "/nopr\n",
                         "/gopr\n",
                         "/nolist\n",
                         "nsub,1,1,1\n",
                         f"time,{t}.\n",
                         "outres,erase\n"
                         "outres,all,none\n"
                         "outres,nsol,all,\n"
                         "outres,rsol,all\n"
                         "outres,etmp,all\n"
                         "solve\n"
                         "/com,****************************************************\n"
                         f"/com,*************** FINISHED SOLVE FOR LS {t} *************\n"]
                    )

        if x == '*get,_wallasol,active,,time,wall\n':
            new_apdl.extend(apdl_dat[i:])

    return new_apdl


def save_results(file, output_folder, nodes, current_index, n_points):
    result = pymapdl_reader.read_binary(file)

    for n in range(n_points):
        displacement_dict = {}

        for t, time in enumerate(range(n * 3, n * 3 + 3)):
            _, displacement = result.nodal_displacement(time)
            displacement_dict[t] = displacement

        pkl_path = os.path.join(output_folder, f"solutionSet_{int(nodes[current_index + n]):06d}.pkl")
        with open(pkl_path, "wb") as pkl_file:
            pickle.dump(displacement_dict, pkl_file)

    del result


def run_apdl(mapdl_location, path, output_apdl, nproc):
    cmd = f"\"{mapdl_location}\" " + \
          f"-p ansys -dis -mpi MSMPI -np {nproc} -lch " + \
          f"-dir \"{path}\" " + \
          "-j file -s read -l en-us -b " + \
          f"-i \"{output_apdl}\" " + \
          f"-o \"{os.path.join(path, 'file.out')}\""
    # "-p ansys -dis -acc nvidia -mpi INTELMPI -np 3 -lch " + \
    subprocess.run(shlex.split(cmd), shell=True)


def main():
    args = parse_args()

    basename = os.path.split(args.input_folder)[1]
    solution_path = args.input_folder

    if args.input_csv:
        input_csv = args.input_csv
    else:
        input_csv = f'{basename}_top_nodes.csv'

    if args.output:
        output_path = args.output
    else:
        output_path = solution_path

    if args.mapdl_location:
        mapdl_location = args.mapdl_location
    else:
        mapdl_location = "C:\\Program Files\\ANSYS Inc\\ANSYS Student\\v211\\ansys\\bin\\winx64\\MAPDL.exe"

    nodes = load_nodes(os.path.join(solution_path, input_csv))
    apdl_dat = load_apdl(os.path.join(solution_path, args.input_filename))

    # create output paths
    pkl_path = os.path.join(output_path, args.subfolder)
    os.makedirs(pkl_path, exist_ok=True)
    output_apdl = os.path.join(output_path, args.output_filename)

    num_blocks = len(nodes) // args.points
    remainder = len(nodes) % args.points

    for block_id in tqdm(range(num_blocks + 1)):
        if block_id == num_blocks:
            n_points = remainder
        else:
            n_points = args.points

        # rebuild apdl file to accommodate more points and minimize building of the model
        new_apdl = rebuild_apdl(apdl_dat, nodes, block_id * args.points, n_points)

        with open(output_apdl, "w") as output_file:
            output_file.writelines(new_apdl)

        while True:
            try:
                path = os.path.join(output_path, f"solutionSet_{int(block_id):06d}")
                if os.path.isdir(path):
                    shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)

                run_apdl(mapdl_location, path, output_apdl, args.nproc)
                save_results(os.path.join(path, "file.rst"), pkl_path, nodes, block_id*args.points, n_points)

                shutil.rmtree(path)

                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt()
            except:
                print("Exception encountered")
                shutil.rmtree(path)
                pass


if __name__ == '__main__':
    main()
