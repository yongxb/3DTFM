import os
import sets
import json

###############################
# Parameters to change
###############################
# The ids of the geometry can be found by selecting the body of the model and running this in the shell below: ExtAPI.SelectionManager.CurrentSelection
geometryId = [857]

# The ids can be found by selecting the respective face and running this in the shell below: ExtAPI.SelectionManager.CurrentSelection
supportFaceId = [861]
topSurfaceId = [858]
yFacesId = [847, 849]
xFacesId = [848, 850]

# set folder to save any generated information
pathName = "E:\\Data\\test_set"

###############################
# Basic housekeeping
###############################
basename = os.path.basename(pathName)

# initiallize mesh object and lists to store node information
mesh = DataModel.MeshDataByName(ExtAPI.DataModel.MeshDataNames[0])

# initiallize an analysis object
analysisOne = Model.Analyses[0]

# Clear old data
with Transaction():
    if Model.NamedSelections != None:
        Model.NamedSelections.Delete()
        
    oldNodalForce = analysisOne.GetChildren[Ansys.ACT.Automation.Mechanical.BoundaryConditions.NodalForce](True)
    for item in oldNodalForce:
        item.Delete()
        
    for item in analysisOne.GetChildren[Ansys.ACT.Automation.Mechanical.BoundaryConditions.FixedSupport](True):
        item.Delete()

###############################
# Get support face
###############################
# select support face
ExtAPI.SelectionManager.ClearSelection()
mySel = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
mySel.Ids = supportFaceId   #These are the IDs of any geometric entities
ExtAPI.SelectionManager.NewSelection(mySel)

supportFaceNodeIds = []
for regionId in supportFaceId:
    supportFace = mesh.MeshRegionById(regionId)
    supportFaceNodeIds.extend(supportFace.NodeIds)

# add a support boundary condition to the bottom of the substrate
support = analysisOne.AddFixedSupport()
support.Location = ExtAPI.SelectionManager.CurrentSelection

###############################
# Get all mesh nodes
###############################
allNodesXYZs = []
for node in mesh.MeshRegionById(geometryId[0]).Nodes:
    allNodesXYZs.append([node.Id, node.X, node.Y, node.Z])

allNodesXYZs.sort(key=lambda x:x[0])

# export node ID number and coordinates to csv for future reference
with open(os.path.join(pathName, "{}_nodes.csv".format(basename)), 'w') as f:
    for nodeId, x, y, z in allNodesXYZs:
        f.write(nodeId.ToString()+","+x.ToString()+","+y.ToString()+","+z.ToString()+"\n")
        
###############################
# Get exterior mesh nodes
###############################
faces = analysisOne.GeoData.Assemblies[0].Parts[0].Bodies[0].Faces
otherNodes = []
topFaceNodeIds = []

for face in faces:
    if face.Id != topSurfaceId[0]:
        otherNodes.extend(mesh.MeshRegionById(face.Id).NodeIds)

topFace = mesh.MeshRegionById(topSurfaceId[0])
topFaceNodeIds.extend(topFace.NodeIds)
topFaceSet = sets.Set(topFaceNodeIds)
otherNodesSet = sets.Set(otherNodes)

sideNodes = topFaceSet.intersection(otherNodesSet)
centerFaceSet = topFaceSet.difference(sideNodes)

output = {}
output_nodes = []
for node in list(centerFaceSet):
    eleIds = mesh.ElementIdsFromNodeIds([node])
    
    eleList = []
    for eleId in eleIds:
        nodes = mesh.NodeIdsFromElementIds([eleId])
        nodes_set = sets.Set(nodes)
        eleNodes = nodes_set.intersection(topFaceSet)
        
        if len(eleNodes) >= 3:
            eleList.append(list(eleNodes))
            
    output[node] = eleList
    
with open(os.path.join(pathName, "elements.json"), "w") as file:
    json.dump(output, file)
    
# export node ID number and coordinates to csv for future reference
with open(os.path.join(pathName, "{}_top_nodes.csv".format(basename)), 'w') as f:
    for nodeId in centerFaceSet:
        node = mesh.NodeById(nodeId)
        f.write(nodeId.ToString()+","+node.X.ToString()+","+node.Y.ToString()+","+node.Z.ToString()+"\n")

###############################
# Set up analysis
###############################
time_values=[]
n=1
total_t = 3*n
analysisOne.AnalysisSettings.PropertyByName("NumberOfSteps").InternalValue=total_t # set number of time-steps (3 for XYZ force)

for t in range(total_t+1):
    time_values.append(Quantity("{} [sec]".format(t)))

meshObj = analysisOne.MeshData
nodeIds = meshObj.NodeIds
blankQuantity = [Quantity ("0 [N]"),Quantity ("0 [N]"),Quantity ("0 [N]")]
initialQuantity = [Quantity ("0 [N]")]

with Transaction():
    selection_info = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.MeshNodes) # initialize a selection info object
    for selected_node in [nodeId]:
        selection_info.Ids.Add(int(selected_node))
        ExtAPI.SelectionManager.NewSelection(selection_info)
        Model.AddNamedSelection()
        selection_info.Ids.Clear()

#####################################
# Generate point forces for 1 node
#####################################
with Transaction():
    for n_index in range(n):
        xQuantity = initialQuantity + blankQuantity*n_index + [Quantity ("0.00000001 [N]"), Quantity ("0 [N]"),Quantity ("0 [N]")] + blankQuantity*(n-n_index-1)
        yQuantity = initialQuantity + blankQuantity*n_index + [Quantity ("0 [N]"), Quantity ("0.00000001 [N]"),Quantity ("0 [N]")] + blankQuantity*(n-n_index-1)
        zQuantity = initialQuantity + blankQuantity*n_index + [Quantity ("0 [N]"), Quantity ("0 [N]"),Quantity ("0.00000001 [N]")] + blankQuantity*(n-n_index-1)
        
        nForce = analysisOne.AddNodalForce()
        nForce.XComponent.Inputs[0].DiscreteValues = time_values
        nForce.YComponent.Inputs[0].DiscreteValues = time_values
        nForce.ZComponent.Inputs[0].DiscreteValues = time_values
        nForce.XComponent.Output.DiscreteValues = xQuantity
        nForce.YComponent.Output.DiscreteValues = yQuantity
        nForce.ZComponent.Output.DiscreteValues = zQuantity
        nForce.Location=Model.NamedSelections.Children[n_index]
        
###############################
# Generate Symmetry in model
###############################
if Model.Symmetry != None:
    Model.Symmetry.Delete()
    
Model.AddSymmetry()
# Add y symmetry
sym1 = Model.Symmetry.AddSymmetryRegion()
sym1.SymmetryNormal = SymmetryNormalType.YAxis
ExtAPI.SelectionManager.ClearSelection()
mySel = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
mySel.Ids = yFacesId   #These are the IDs of any geometric entities
ExtAPI.SelectionManager.NewSelection(mySel)
sym1.Location = mySel
sym1.Type = SymmetryRegionType.Symmetric

# Add x symmetry
sym2 = Model.Symmetry.AddSymmetryRegion()
sym2.SymmetryNormal = SymmetryNormalType.XAxis
ExtAPI.SelectionManager.ClearSelection()
mySel = ExtAPI.SelectionManager.CreateSelectionInfo(SelectionTypeEnum.GeometryEntities)
mySel.Ids = xFacesId   #These are the IDs of any geometric entities
ExtAPI.SelectionManager.NewSelection(mySel)
sym2.Location = mySel
sym2.Type = SymmetryRegionType.Symmetric

# write out apdl file for the next step
analysisOne.WriteInputFile(os.path.join(pathName, "ds.dat"))
