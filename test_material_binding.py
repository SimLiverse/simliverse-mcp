from pxr import Usd, UsdGeom, UsdShade

stage = Usd.Stage.CreateInMemory()
prim = UsdGeom.Cube.Define(stage, '/Cube').GetPrim()
material_prim = stage.DefinePrim('/Material')
material = UsdShade.Material(material_prim)

try:
    binding = UsdShade.MaterialBindingAPI(prim)
    print('binding.GetPrim().IsValid():', binding.GetPrim().IsValid())
    binding.Bind(material)
    print('Bind with just constructor succeeded')
except Exception as e:
    print('Error with constructor:', e)

try:
    binding_apply = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_apply.Bind(material)
    print('Bind with Apply succeeded')
except Exception as e:
    print('Error with Apply:', e)
