# Inverse problem and Tikhonov regularization

Python scripts to solve for forces by treating the system as a linear inverse problem.

***
## Overview
Currently, the system (calculated displacements from the finite element model) and the output (vectors from 3D 
optical flow) are known. We can thus solve for the forces by viewing this as a linear inverse problem. This is 
achieved via the singular vector decomposition method. Also, as the problem is ill-defined, regularization is needed.
Tikhonov regularization is performed with the regularization parameter being determined using the L-curve method.

## Required packages
The following packages are required. Please ensure that they are installed using either pip or conda.
- pandas
- scipy
- numba
- numpy
- matplotlib

***

## Usage
### Prerequisites
Please ensure that the [required packages](#required-packages) are installed. 

### Folder setup
Please also ensure that you have the following files in the input folder:
- CSV containing nodal information
- Interpolated displacements from ANSYS. Please load the forces obtain from ([vector processing](../5_Vector_processing)) and save 
  the nodal displacements from ANSYS. 
- Folder containing the FEM results from the nodal forces

### Arguments
| Argument       | Type       | Description                                                                                                                 |
|----------------|------------|-----------------------------------------------------------------------------------------------------------------------------|
| input_folder   | *Required* | Parent folder containing all the needed folder/files                                                                        |
| --displacement | Optional   | .txt file containing the comma separated interpolated displacement values from ANSYS. Default:interpolated_displacement.txt |
| --subfolder    | Optional   | Subfolder inside \<output> that is used to store the results. Default: pkl                                                  |
| --input-csv    | Optional   | .csv file containing the nodal information. Default: \<folder_basename>_top_nodes.csv                                       | 
| --svd-file     | Optional   | Name of input ANSYS .dat model. Default: ds.dat                                                                             |
| --nodal-force  | Optional   | Nodal force used in FEM model. Default: 1e-8                                                                                |
| --pixel-um     | Optional   | Nodal force used in FEM model. Default: 1e-8                                                                                |
| --save-plot    | Optional   | File name to save l curve plot. If set to None, the plot is not generated. Default: l_curve.png                             |
| --save-forces  | Optional   | File name to save force solution. Default: forces.csv                                                                       |

### Simple example
If you have been following the pipeline, this can be run using the code below after replacing ```<input_folder>``` with 
your folder containing the ANSYS model and nodal CSV file. Please note the use of "\\" as the separator. 

```
python inverse_problem.py "<input_folder>"
```

Example
```
python inverse_problem.py "E:\\Data\\test_set"
```

### Changing the nodal force used
If a different nodal force is used in the ANSYS model, this can be changed as follows

```
python inverse_problem.py "<input-folder>" --nodal-force "<nodal-force>"
```

Example
```
python inverse_problem.py "E:\\Data\\test_set" --nodal-force "1e-8"
```
### Changing the pixel to µm value
If a different microscope objective is used to acquire the image, the pixel to µm value needs to be changed

```
python inverse_problem.py "<input-folder>" --pixel-um "<pixel-um>"
```

Example
```
python inverse_problem.py "E:\\Data\\test_set" --pixel-um 0.275
```
***
## How to cite

