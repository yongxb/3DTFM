# Contact pressure

Python script to calculate the contact pressure from the discrete nodal forces.

***
## Overview
Currently, the system (calculated displacements from the finite element model) and the output (vectors from 3D 
optical flow) are known. We can thus solve for the forces by viewing this as a linear inverse problem. This is 
achieved via the singular vector decomposition method. Also, as the problem is ill-defined, regularization is needed.
Tikhonov regularization is performed with the regularization parameter being determined using the L-curve method.

## Required packages
The following packages are required. Please ensure that they are installed using either pip or conda.
- pandas
- numpy

***

## Usage
### Prerequisites
Please ensure that the [required packages](#required-packages) are installed. 

### Folder setup
Please also ensure that you have the following files in the input folder:
- .json containing the element information
- .csv containing nodal information for the top surface
- .csv containing nodal information for all nodes
- .csv containing the calculated forces

### Arguments
| Argument           | Type       | Description                                                                                                    |
|--------------------|------------|----------------------------------------------------------------------------------------------------------------|
| input_folder       | *Required* | Parent folder containing all the needed folder/files                                                           |
| --element-filename | Optional   | .json file containing the elements in the ANSYS model. Default: elements.json                                  |
| --nodes-csv        | Optional   | .csv file containing the nodal information for all nodes. Default: \<folder_basename>_nodes.csv                |
| --top-nodes-csv    | Optional   | .csv file containing the nodal information for the top surface nodes. Default: <folder_basename>_top_nodes.csv | 
| --forces           | Optional   | .csv file containing the solved forces. Default: forces.csv                                                    |
| --output-csv       | Optional   | Output .csv file containing the contact pressure at each node. Default: contact_pressure.csv                   |
| --width            | Optional   | Specifies the width between the nodes. Default: 0.02                                                           |
| --pixel-um         | Optional   | Pixel to µm scaling factor. Default: 0.275                                                                     |

### Simple example
If you have been following the pipeline, this can be run using the code below after replacing ```<input_folder>``` with 
your folder containing the required files as in [folder setup](#folder-setup). Please note the use of "\\" as 
the separator. 

```
python contact_pressure.py "<input_folder>"
```

Example
```
python contact_pressure.py "E:\\Data\\test_set"
```

### Changing the width
The ```width``` parameter is used to specify the distance between adjacent nodes. This is used to generate a grid of 
points with equal distances between them along the surface of the curvature. Note that this value might need to be 
tweaked slightly (e.g. 0.0195 or 0.0205) as meshing does nots result in every point having an exact distance of 0.02 
from every neighbouring point.

```
python contact_pressure.py "<input-folder>" --width <width>
```

Example
```
python contact_pressure.py "E:\\Data\\test_set" --width 0.02
```

### Specifying the pixel to µm setting
If the image has a different pixel to µm setting, it can be specified as such
```
python contact_pressure.py "<input-folder>" --pixel_um <pixel_um>
```

Example
```
python contact_pressure.py "E:\\Data\\test_set" --pixel_um 0.275
```
***
## How to cite

