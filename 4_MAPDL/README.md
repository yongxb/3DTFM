# MAPDL module

Python scripts to run the ANSYS model and obtain the displacements for each nodal force. 

***
## Overview
In the previous steps, we have setup an ANSYS model with the nodal forces for a single node. While we can use a 
python script to cycle through all the nodes in ANSYS individually, it will be faster to modify the APDL file to 
analyze multiple points as a time series. This saves on the time required to build the model. This also opens up the 
option to compress the output using pickle as we can avoid the use of IronPython found in ANSYS (and its lack of 
support for many Python packages).

## Required packages
The following packages are required. Please ensure that they are installed using either pip or conda.
- pandas
- PyMAPDL
- tqdm

***

## Usage
### Prerequisites
Please ensure that the [required packages](#required-packages) are installed. Please also ensure that a version of 
ANSYS is properly installed. This project used the ANSYS Student 2021 R1 bundle. 

### Folder setup
Please also ensure that you have the following files in the input folder:
- CSV containing nodal information
- ANSYS .dat model

### Arguments
| Argument          | Type       | Description                                                                                                                        |
|-------------------|------------|------------------------------------------------------------------------------------------------------------------------------------|
| input-folder      | *Required* | Input folder containing the ANSYS .dat model and CSV with nodal information                                                        |
| --input-csv       | Optional   | CSV file containing the nodal information. Default: \<folder_basename>_top_nodes.csv                                               | 
| --input-filename  | Optional   | Name of input ANSYS .dat model. Default: ds.dat                                                                                    |
| --output-filename | Optional   | Name of temporary ANSYS .dat model. Default: ds_temp.dat                                                                           |
| --output          | Optional   | Output folder to store the results and temporary files. Default: \<folder>                                                         |
| --subfolder       | Optional   | Subfolder inside \<output> that is used to store the results. Default: pkl                                                         |
| --points          | Optional   | Number of points in each batch. Default: 100                                                                                       |
| --nproc           | Optional   | Number of processors to use in Ansys Mechanical APDL. Default: 4                                                                   |
| --mapdl-location  | Optional   | Location of MAPDL.exe. Default: "C:\\\\Program Files\\\\ANSYS Inc\\\\ANSYS Student\\\\v211\\\\ansys\\\\bin\\\\winx64\\\\MAPDL.exe" |

### Simple example
If you have been following the pipeline, this can be run using the code below after replacing \<input-folder> with 
your folder containing the ANSYS model and nodal CSV file. Please note the use of "\\" as the separator. 

```
python run_model.py "<input-folder>"
```

Example
```
python run_model.py "E:\\Data\\test_set"
```

### Changing the MAPDL location
If you are not using ANSYS Student 2021 R1 or if it is not installed in the default location, the MAPDL.exe location 
needs to be specified. This can be done as shown below. Please note the use of "\\" as the separator. 

```
python run_model.py "<input-folder>" --mapdl-location="<mapdl-location>"
```

Example for ANSYS Student 2021 R2:
```
python run_model.py "E:\\Data\\test_set" --mapdl-location="C:\\Program Files\\ANSYS Inc\\ANSYS 
Student\\v212\\ansys\\bin\\winx64\\MAPDL.exe"
```

### Changing the number of processors
If you have a license that allows for more processor to be used, this can be specified using the --nproc option. 
Note that for the ANSYS Student license, this is limited to a maximum of 4 processor.

```
python run_model.py "<input-folder>" --nproc="<num_logical_processors>"
```

Example
```
python run_model.py "E:\\Data\\test_set" --nproc=4
```

### Changing the number of nodal points in each batch
The number of nodal points determine the number of times that the model needs to be setup in ANSYS. Setting a low value 
would result in a long time to completion as the model needs to be setup more often. Setting too high a value could 
also lead to quite a bit of additional time should a batch fail for any reason and needs to be repeated. Note that 
there is a diminishing margin of return as this value increases. 100 nodal points seems to be a good compromise.

```
python run_model.py "<input-folder>" --points="<num_points>"
```

Example
```
python run_model.py "E:\\Data\\test_set" --points=100
```
***
## How to cite

