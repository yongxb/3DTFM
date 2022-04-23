# Visualization module

Python script to visualize the calculated forces.

***
## Overview


## Required packages
The following packages are required. Please ensure that they are installed using either pip or conda.
- pandas
- numpy
- mayavi

***

## Usage
### Prerequisites
Please ensure that the [required packages](#required-packages) are installed. 

### Folder setup
Please also ensure that you have the following files in the input folder:
- .csv containing the calculated contact pressure

### Arguments
| Argument      | Type       | Description                                                                                       |
|---------------|------------|---------------------------------------------------------------------------------------------------|
| input_file    | *Required* | .csv file containing the forces                                                                   |
| visualization | Optional   | Type of visualization. See [visualizations](#visualizations) for list of supported visualizations |
| flatten       | Optional   | Whether to show the forces in its 3D space or as a flattened surface. Default: False              |
| z-pos         | Optional   | Whether to plot z position on a wireframe grid. Only valid if --flatten is True. Default: False   |

### Example
The forces can be plotted for visualization using the following code below replacing ```<input_file>``` with the 
path to the .csv file containing the forces.
```
python visualization.py "<input_file>"
```

Example
```
python visualization.py "E:\\Data\\test_set\\contact_pressure.csv"
```

### Visualizations
The following visualizations are supported 

| Value        | Description                                                                                     |
|--------------|-------------------------------------------------------------------------------------------------|
| forces       | Visualize calculated forces based on its 3D location                                            |
| orthogonal   | Visualize the orthogonal x, y, and z force components relative to the tangent plane of the node |
| orthogonal_x | Visualize the orthogonal x force component relative to the tangent plane of the node            |
| orthogonal_y | Visualize the orthogonal y force component relative to the tangent plane of the node            |
| orthogonal_z | Visualize the orthogonal z force component relative to the tangent plane of the node            |

They can be selected as follows:

```
python visualization.py "<input_file>"
```

Example
```
python visualization.py "E:\\Data\\test_set\\contact_pressure.csv" -v forces
```
***

### Flattening the surface 
If the surface curvature can be flattened, the forces can be visualized on a 2D plane.

Example
```
python visualization.py "E:\\Data\\test_set\\contact_pressure.csv" -v orthogonal --flatten True
```

### Add the z position visualization
In addition, the z position can be plotted by setting the ```z-pos``` value to True. This plots a wireframe grid 
that is color coded with the z values.

Example
```
python visualization.py "E:\\Data\\test_set\\contact_pressure.csv" -v orthogonal --flatten True --z-pos True
```
***
## How to cite

