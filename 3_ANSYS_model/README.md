# ANSYS model

Python scripts to setup the model in ANSYS and save the .dat apdl file for the next step.

***
## Overview
This setups the ANSYS model with the correct boundary conditions and outputs the APDL file as well as another file 
containing the nodal information.

***

## Usage
### Prerequisites
This has to be run within ANSYS Mechanical. Please load it via Automation >> Scripting.

The following also needs to be done before the script can be run.
- The material property of the model should be set properly.
- An appropriate mesh needs to be generated for the model.

### Obtaining the id of objects
The id of the various geometry and faces can be obtained by selecting the object and running the following line in the 
shell of ANSYS mechanical:
```
ExtAPI.SelectionManager.CurrentSelection
```

### Getting the id of the geometry
The id of the geometry needs to be specified.
```
geometryId = [<id of geometry>]
```
Example
```
geometryId = [857]
```

### Getting the id of the faces
The id of the following faces also needs to be obtained and specified in the file:

| Id            | Example                     | Remarks                                                          |
|---------------|-----------------------------|------------------------------------------------------------------|
| supportFaceId | ```supportFaceId = [861]``` | Specifies which faces should be the fixed support                |
| topSurfaceId  | ```topSurfaceId = [858]```  | Specifies the faces where nodal forces will be generated         |
| yFacesId      | ```yFacesId = [847, 849]``` | Specifies the faces which should be symmetrical along the y axis |
| xFacesId      | ```xFacesId = [848, 850]``` | Specifies the faces which should be symmetrical along the x axis |

### Specify the output 
The output folder, in which the APDL .dat file and the nodal information will be placed, can also be specified.
```
pathName = "<folder path>"
```
Example
```
pathName = "E:\\Data\\test_set"
```


***
## How to cite

