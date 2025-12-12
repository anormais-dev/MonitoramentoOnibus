# fix_orientation_and_export.py
# Uso: blender -b -P fix_orientation_and_export.py
import bpy, os, math
from mathutils import Euler, Vector

MODEL_IN = "3dModelBus.glb"
MODEL_OUT = "3dModelBus_aligned.glb"

def clean_scene_except_import():
    # Remove tudo do scene (só para garantir ambiente limpo)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)

def import_model(path):
    ext = path.lower().split('.')[-1]
    if ext in ("glb", "gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == "fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == "obj":
        bpy.ops.import_scene.obj(filepath=path)
    else:
        raise RuntimeError("Formato não suportado: " + ext)

def get_imported_objects():
    return [o for o in bpy.context.scene.objects if o.type == 'MESH']

def compute_center(objects):
    coords = []
    for o in objects:
        for v in o.bound_box:
            world_v = o.matrix_world @ Vector(v)
            coords.append(world_v)
    if not coords:
        return Vector((0,0,0))
    minc = Vector((min(c.x for c in coords), min(c.y for c in coords), min(c.z for c in coords)))
    maxc = Vector((max(c.x for c in coords), max(c.y for c in coords), max(c.z for c in coords)))
    return (minc + maxc) / 2.0

def main():
    cwd = os.getcwd()
    in_path = os.path.join(cwd, MODEL_IN)
    out_path = os.path.join(cwd, MODEL_OUT)

    if not os.path.exists(in_path):
        raise FileNotFoundError("Arquivo não encontrado: " + in_path)

    # Clean scene and import
    clean_scene_except_import()
    import_model(in_path)

    imported = get_imported_objects()
    if not imported:
        raise RuntimeError("Nenhum mesh importado do GLB.")

    # Compute center
    center = compute_center(imported)

    from mathutils import Matrix

    # Apply an initial -90deg rotation around X to each imported object (this "deita" the bus)
    rot = Euler((math.radians(-90.0), 0.0, 0.0), 'XYZ')
    Rmat = rot.to_matrix().to_4x4()  # rotation matrix 4x4

    for o in imported:
        # multiply rotation matrix by object's world matrix to rotate in world-space
        mw = o.matrix_world.copy()
        o.matrix_world = Rmat @ mw

    # Select all imported objects and apply transforms (freeze rotation)
    bpy.ops.object.select_all(action='DESELECT')
    for o in imported:
        o.select_set(True)
    bpy.context.view_layer.objects.active = imported[0]

    # Translate objects so center becomes origin (avoid visual jump when parented)
    for o in imported:
        mw = o.matrix_world.copy()
        mw.translation = mw.translation - center
        o.matrix_world = mw

    # Apply rotation now (freeze rotation)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Optional: join objects under one parent empty (keeps scene organized)
    root = bpy.data.objects.new("MODEL_ROOT", None)
    bpy.context.collection.objects.link(root)
    root.location = (0.0, 0.0, 0.0)
    for o in imported:
        o.parent = root
        try:
            o.matrix_parent_inverse = root.matrix_world.inverted()
        except Exception:
            bpy.context.view_layer.update()
            o.matrix_parent_inverse = root.matrix_world.inverted()

    # Export aligned GLB
    bpy.ops.export_scene.gltf(
        filepath=out_path,
        export_format='GLB',
        export_materials='EXPORT',
        export_animations=False
    )
    print("Exported aligned GLB to:", out_path)

if __name__ == "__main__":
    main()
