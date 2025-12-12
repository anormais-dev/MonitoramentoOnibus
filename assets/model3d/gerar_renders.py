# gerar_renders.py
# Blender headless render script
# Requirements: blender 2.8+ (recommended 3.x)
# Usage: blender -b -P gerar_renders.py

import bpy, os, math, random
from mathutils import Vector, Euler

# ---------------- CONFIG ----------------
MODEL_FILE = "3dModelBus_aligned.glb"
HDRI_FILE = "hdri.hdr"
OUT_DIR = "renders"
N_IMAGES = 100
RES_X = 1024
RES_Y = 576
SAMPLES = 64            # ajuste se quiser qualidade maior (mais tempo)
USE_CYCLES = True
RADIUS_SCALE = 2.2      # câmera distância = bbox_max_dim * RADIUS_SCALE
MIN_ELEV = 5            # degrees
MAX_ELEV = 25           # degrees
# ----------------------------------------

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    # remove meshes data
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block, do_unlink=True)

def import_model(path):
    ext = path.lower().split('.')[-1]
    if ext in ("glb","gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == "fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == "obj":
        bpy.ops.import_scene.obj(filepath=path)
    else:
        raise RuntimeError("Formato não suportado: " + ext)

def get_objects_from_import():
    return [o for o in bpy.data.objects if o.type in ('MESH','CURVE','SURFACE','META')]

def create_parent_empty(name="MODEL_ROOT"):
    empty = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(empty)
    return empty

def parent_objects_to(obj_list, parent):
    for o in obj_list:
        o.select_set(True)
        o.parent = parent

def compute_bounding_radius(obj_list):
    # compute bounding box extremes in world coords
    coords = []
    for o in obj_list:
        for v in o.bound_box:
            world_v = o.matrix_world @ Vector(v)
            coords.append(world_v)
    if not coords:
        return 1.0, Vector((0,0,0))
    minc = Vector((min(c.x for c in coords),
                   min(c.y for c in coords),
                   min(c.z for c in coords)))
    maxc = Vector((max(c.x for c in coords),
                   max(c.y for c in coords),
                   max(c.z for c in coords)))
    center = (minc + maxc) / 2.0
    diag = (maxc - minc).length
    radius = diag / 2.0
    return max(radius, 0.1), center

def create_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.lens = 35  # mm
    return cam

def create_aux_light():
    # Add an area light to give some rim/contrast (intensity will vary)
    light_data = bpy.data.lights.new(name="AreaLight", type='AREA')
    light = bpy.data.objects.new(name="AreaLight", object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = (3, -3, 4)
    light.data.size = 3.0
    light.data.energy = 500.0
    return light

def setup_world_hdri(hdri_path):
    # use nodes for world with mapping so we can rotate HDRI
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    # nodes
    node_tex = nodes.new(type="ShaderNodeTexEnvironment")
    node_tex.image = bpy.data.images.load(hdri_path)
    node_tex.location = (-300,0)
    node_map = nodes.new(type="ShaderNodeMapping")
    node_coord = nodes.new(type="ShaderNodeTexCoord")
    node_bg = nodes.new(type="ShaderNodeBackground")
    node_out = nodes.new(type="ShaderNodeOutputWorld")
    # links
    links.new(node_coord.outputs["Generated"], node_map.inputs["Vector"])
    links.new(node_map.outputs["Vector"], node_tex.inputs["Vector"])
    links.new(node_tex.outputs["Color"], node_bg.inputs["Color"])
    links.new(node_bg.outputs["Background"], node_out.inputs["Surface"])
    # return mapping node to adjust rotation later
    return node_map, node_bg

def ensure_cycles(use_gpu=True):
    if not USE_CYCLES:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        return
    bpy.context.scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences
    # Try to set GPU if available
    # user can adjust if needed in Blender GUI
    devices = cycles_prefs.get_devices()
    # leave default selection; setting exact GPU programmatically is brittle across versions
    bpy.context.scene.cycles.samples = SAMPLES

def render_sequence(model_root, cam, mapping_node, area_light, out_dir):
    scene = bpy.context.scene
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # compute bounding radius and center
    imported = get_objects_from_import()

    for o in imported:
        # aplica rotação local
        o.rotation_euler = Euler((
            math.radians(-90),   # gira o modelo para deitar
            0,
            0
        ), 'XYZ')

    # aplica transforms para fixar a rotação no mesh
    bpy.ops.object.select_all(action='DESELECT')
    for o in imported:
        o.select_set(True)
    bpy.context.view_layer.objects.active = imported[0]
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    radius, center = compute_bounding_radius(imported)
    # move parent to center and set object's origin
    model_root.location = -center
    # compute camera radius
    cam_radius = radius * RADIUS_SCALE

    for i in range(N_IMAGES):
        # camera spherical coordinates
        az = 2.0 * math.pi * (i / float(N_IMAGES)) + random.uniform(-0.05, 0.05)
        elev = math.radians(random.uniform(MIN_ELEV, MAX_ELEV))
        x = cam_radius * math.cos(elev) * math.cos(az)
        y = cam_radius * math.cos(elev) * math.sin(az)
        z = cam_radius * math.sin(elev) + radius * 0.1

        cam.location = (x, y, z)
        # point camera to origin (0,0,0)
        cam.rotation_euler = (Euler((0,0,0),"XYZ"))
        cam_constraint = cam.constraints.get("Track")
        if not cam_constraint:
            tc = cam.constraints.new(type='TRACK_TO')
            tc.target = model_root
            tc.track_axis = 'TRACK_NEGATIVE_Z'
            tc.up_axis = 'UP_Y'

        # rotate the model a bit independently
        model_root.rotation_euler = Euler((
            math.radians(random.uniform(-3, 3)),   # small random tilt X
            math.radians(random.uniform(-2, 2)),   # small random tilt Y
            math.radians(random.uniform(-30, 30))  # small random yaw variation only (not az)
        ), 'XYZ')

        # vary HDRI rotation on Z axis via mapping node
        rot_z = random.uniform(0.0, 2.0*math.pi)
        # vary HDRI rotation on Z axis via mapping node (robust across Blender versions)
        try:
            # API normal: Mapping node has input named 'Rotation'
            mapping_node.inputs['Rotation'].default_value[2] = rot_z
        except Exception:
            try:
                # fallback por índice (algumas versões usam índice 2)
                mapping_node.inputs[2].default_value[2] = rot_z
            except Exception:
                # se tudo falhar, ignora — não crítico
                pass


        # vary environmental intensity by scaling background node strength if present
        # if background node exists, adjust its strength via node inputs (if not, skip)
        try:
            # background node is mapping_node's parent links -> node_bg we returned earlier potentially
            # we'll adjust scene.world.node_tree.nodes["Background"].inputs[1]
            bg_node = scene.world.node_tree.nodes.get("Background")
            if bg_node:
                bg_node.inputs[1].default_value = random.uniform(0.6, 1.6)  # strength
        except Exception:
            pass

        # vary area light energy a bit
        area_light.data.energy = random.uniform(200.0, 1200.0)

        # render
        fname = os.path.join(out_dir, f"frame_{i:04d}.png")
        scene.render.filepath = fname
        print(f"[render] {i+1}/{N_IMAGES} -> {fname}")
        bpy.ops.render.render(write_still=True)

def main():
    # cleanup
    clean_scene()
    # import model
    model_path = os.path.join(os.getcwd(), MODEL_FILE)
    if not os.path.exists(model_path):
        raise RuntimeError("Modelo não encontrado: " + model_path)
    import_model(model_path)

   # ---------- centraliza, corrige orientação e parenta ----------
    imported = get_objects_from_import()

    # calcula bounding box world-space
    radius, center = compute_bounding_radius(imported)

    # calcula extents por eixo (world-space)
    coords = []
    for o in imported:
        for v in o.bound_box:
            world_v = o.matrix_world @ Vector(v)
            coords.append(world_v)
    minc = Vector((min(c.x for c in coords),
                min(c.y for c in coords),
                min(c.z for c in coords)))
    maxc = Vector((max(c.x for c in coords),
                max(c.y for c in coords),
                max(c.z for c in coords)))
    extents = maxc - minc  # Vector(x_extent, y_extent, z_extent)

    # identifica indice do eixo com menor extent (deve ser 'altura' do ônibus)
    min_idx = 0
    if extents.y < extents.x and extents.y <= extents.z:
        min_idx = 1
    elif extents.z < extents.x and extents.z < extents.y:
        min_idx = 2
    else:
        min_idx = 0

    # cria empty root na origem
    model_root = create_parent_empty("MODEL_ROOT")
    model_root.location = (0.0, 0.0, 0.0)

    # rotaciona o root para que o eixo com menor extent passe a ser Z
    # mapeamento:
    # - se menor axis é X (0) -> rotacionar -90 deg em Y (X -> Z)
    # - se menor axis é Y (1) -> rotacionar +90 deg em X (Y -> Z)
    # - se menor axis é Z (2) -> sem rotação
    import math
    from mathutils import Euler
    if min_idx == 0:
        model_root.rotation_euler = Euler((0.0, -math.pi/2.0, 0.0), 'XYZ')
    elif min_idx == 1:
        model_root.rotation_euler = Euler((math.pi/2.0, 0.0, 0.0), 'XYZ')
    else:
        model_root.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')

    # Traslada os objetos para que o centro do modelo vire a origem,
    # aplicando compensação na matrix_world antes de parentar.
    for o in imported:
        mw = o.matrix_world.copy()
        mw.translation = mw.translation - center
        o.matrix_world = mw

    # parenta os objetos ao root e ajusta matrix_parent_inverse
    for o in imported:
        o.parent = model_root
        try:
            o.matrix_parent_inverse = model_root.matrix_world.inverted()
        except Exception:
            bpy.context.view_layer.update()
            o.matrix_parent_inverse = model_root.matrix_world.inverted()
    # -------------------------------------------------------------



    # create camera & light
    cam = create_camera()
    area_light = create_aux_light()

    # setup world with HDRI
    hdri_path = os.path.join(os.getcwd(), HDRI_FILE)
    if not os.path.exists(hdri_path):
        raise RuntimeError("HDRI não encontrado: " + hdri_path)
    mapping_node, bg_node = setup_world_hdri(hdri_path)

    # cycles setup
    ensure_cycles()

    # render sequence
    out_dir = os.path.join(os.getcwd(), OUT_DIR)
    render_sequence(model_root, cam, mapping_node, area_light, out_dir)
    print("Renderização completa. Arquivos em:", out_dir)

if __name__ == "__main__":
    main()
