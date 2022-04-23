# Python Script, API Version = V19
# Insert From File
file_path = "E:\\data\\test_set\\test_set_pointcloud.txt"
DocumentInsert.Execute(file_path)
# EndBlock

# Get the root part
part = GetRootPart()
print part.GetName()

# Get the width and length
curves = part.Curves
width = 0
length = 0
reflection_distance = 0
for curve in curves:
    end_point = curve.EvalEnd().Point
    width = max(width, end_point[0]*1000)
    length = max(length, end_point[1]*1000)

# Create Blend
selection = Selection.Create(curves)
options = LoftOptions()
options.GeometryCommandOptions = GeometryCommandOptions()
Loft.Create(selection, None, options)
# EndBlock

# Set Sketch Plane
sectionPlane = Plane.PlaneXY
ViewHelper.SetSketchPlane(sectionPlane)
# EndBlock

# Set New Sketch
SketchHelper.StartConstraintSketching()
# EndBlock

# Sketch Rectangle
point1 = Point2D.Create(MM(-reflection_distance),MM(-0))
point2 = Point2D.Create(MM(width+reflection_distance),MM(-0))
point3 = Point2D.Create(MM(width+reflection_distance),MM(length+0))
SketchRectangle.Create(point1, point2, point3)
# EndBlock

# Create Blend
selection = Selection.Create(part.DatumPlanes[0].Curves)
options = LoftOptions()
options.GeometryCommandOptions = GeometryCommandOptions()
Loft.Create(selection, None, options)
# EndBlock

# Create Blend
selection = BodySelection.Create(part.Bodies) #Body1, Body2)
options = LoftOptions()
options.GeometryCommandOptions = GeometryCommandOptions()
result = Loft.Create(selection, None, options)
# EndBlock

# Exit sketch editing mode
Sketch3D.Toggle3DSketchMode()