# Profile generation

Python script to obtain the profile of the structure from fluorescent bead images.

***
## Overview
A 3D representation of the structure needs to be generated. This can be done by obtaining the alpha shape that 
surround the beads. First, the location of the beads are obtained. Next, the locations are organized into 
sub-volumes in order to generate multiple profiles along the structure. The profiles are saved in a .txt file that 
can be loaded into SpaceClaim to generate the 3D model.

## Required packages
The following packages are required. Please ensure that they are installed using either pip or conda.
- pandas
- numpy
- pykdtree
- alphashape
- scikit-image
- scipy

***

## Usage
### Prerequisites
Please ensure that the [required packages](#required-packages) are installed.

### Example
Please see [profile.ipynb](profile.ipynb) for an example of how to use this. 

***
## How to cite

