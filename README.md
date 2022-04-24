# 3DTFM pipeline

This repository contains the various codes required to setup the ANSYS model and solve for the 3D forces from the 
model. Please also see [opticalflow3D](https://gitlab.com/xianbin.yong13/opticalflow3d) for the companion python 
package to obtain the vector displacements between two image volumes.

***
## Components
0. Displacement using 3D optical flow ([opticalflow3D](https://gitlab.com/xianbin.yong13/opticalflow3d))
1. 3D profile generation ([profile generation](1_Profile_generation))
2. Creating 3D representation using ANSYS SpaceClaim ([SpaceClaim](2_Spaceclaim))
3. Setting up ANSYS model ([ANSYS model](3_ANSYS_model/README.md))
4. Running FEM analysis ([MAPDL](4_MAPDL/README.md))
5. Processing and filtering vectors ([vector processing](5_Vector_processing))
6. Solving for forces ([inverse problem](6_Inverse_problem))
7. Average the forces ([contact pressure](7_Contact_pressure))
8. Visualization of forces ([visualization](8_Visualization))

***
## How to cite