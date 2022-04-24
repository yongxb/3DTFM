# Vector processing
Python script to remove any global movements (such as drifting) inherent in the image volumes

***
## Overview
It is unlikely that there is no movement between the acquisition of the two image volume. Thus, this script remove 
these movements by using the median displacement values in a region which should be static. In addition, it also uses a 
line profile based method to correct for any changes to the gel height in the z direction. Lastly, this script also 
sub-samples the vectors and output it in a format that can be imported into ANSYS.

## Required packages
The following packages are required. Please ensure that they are installed using either pip or conda.
- cupy
- numpy
- matplotlib

***

## Usage
### Prerequisites
Please ensure that the [required packages](#required-packages) are installed. 

### Folder setup
Please also ensure that you have the following files in the input folder:
- .npz file containing the vectors obtained from 3D optical flow
- .npz file containing the errors obtained from 3D optical flow (if available)
- .npz file containing the volume mask (if available)

### Arguments
| Argument         | Type       | Description                                                                                                         |
|------------------|------------|---------------------------------------------------------------------------------------------------------------------|
| input_folder     | *Required* | Parent folder containing all the needed folder/files                                                                |
| --vector-file    | Optional   | .npz file containing the vectors obtained from 3D optical flow. Default: vectors.npz                                |
| --error-file     | Optional   | .npz file containing the errors obtained from 3D optical flow. Default: None                                        |
| --vol-file       | Optional   | .npz file containing the volume mask. Default: None                                                                 | 
| --z-volume-start | Optional   | Specifies the z start position to define the volume to calculate the median values from. Default: 25                |
| --z-volume-end   | Optional   | Specifies the z end position to define the volume to calculate the median values from. Default: 50                  |
| --edge-width     | Optional   | Specifies the distance from the edge to ignore during median value calculation. Default: 50                         |
| --z-correction   | Optional   | Specifies the z correction method. Supported values are \[median, profile]. Default: median                         |
| --profile-pos    | Optional   | Specifies the x position of the line with which to obtain the profile. Required if "profile" is used. Default: None |
| --profile-width  | Optional   | Specifies the width of the line profile in the x direction. Required if "profile" is used. Default: None            |
| --profile-z-end  | Optional   | Specifies the height of the profile from the base of the structure. Default: None                                   |
| --sigma          | Optional   | Sigma of Gaussian kernel used during downsampling. Default: 1                                                       |
| --step           | Optional   | Downsampling step size. Default: 4                                                                                  |
| --output-file    | Optional   | Output .txt file containing forces that can be imported into ANSYS. Default: ansys_displacement.txt                 |

### Example
If the 3D Farneback algorithm is used, the vectors can be filtered using both the calculated errors and the volume 
mask as follows.

Example
```
python vector_processing.py "E:\\Data\\test_set" --vector-file vectors.npz --error-file errors.npz --vol-file vol.
npz
```

### Setting up the profile z-correction method
To set up the profile method, please identify a region within the volume which should be static. For example, the 
hill regions could be used for a wavy substrate. For example, the x location of the hill was at 790 px from the left 
of the volume). A z height of 600 px was chosen as it leaves some buffer to the surface to remove any influence that 
the cells have on this correction method. 

Example
```
python vector_processing.py "E:\\Data\\test_set" --vector-file vectors.npz --error-file errors.npz --vol-file vol.
npz --z-correction profile --profile-pos 790 --profile-width 20 --profile-z-end 600
```

***
## How to cite

