"""
Arma public/models/boxer_mpfb2.glb con MPFB2 dentro de Blender, sin interfaz.

POR QUE EXISTE
  El avatar del perfil (specs/cambios-27-10/01-avatar-mpfb2.md) carga un GLB con contrato fijo:
  mallas con nombre, rig "Game engine" de MPFB2, PBR y un presupuesto de triangulos y de peso.
  Exportarlo a mano desde Blender es una tarde de clicks que nadie puede repetir igual. Esto lo
  arma entero desde un script, asi el GLB se rehace con un comando.

  El paquete de pieles de MakeHuman baja a ~8 KB/s (280 MB, horas): la piel no sale de ahi. Se
  usa la piel LAYERED de MPFB2, que es procedural, y se hornea a texturas con Cycles.

QUE HACE
  1. Humano MPFB2 (varon adulto, musculoso, magro) y rig game_engine
  2. Ojos simples sobre los helpers de MPFB2 (sin el paquete de assets no hay ojos)
  3. Saca los helpers y las manos (las tapan los guantes)
  4. Short de boxeo desde las caras de la cadera, con la piel de abajo borrada
  5. Guantes modelados con metaballs, remallados, con cuero horneado y abrojo
  6. Hornea piel (color, rugosidad con sudor, normal con poros) y exporta GLB

USO
  ~/blender/blender-4.2.23-linux-x64/blender -b --python scripts/avatar_mpfb2.py -- \\
      --salida public/models/boxer_mpfb2.glb [--textura 2048] [--previa previa.png]
  Despues: node scripts/validar-glb.mjs public/models/boxer_mpfb2.glb
"""
import argparse
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

from bl_ext.blender_org.mpfb.services.humanservice import HumanService
from bl_ext.blender_org.mpfb.services.targetservice import TargetService

CUERPO = "Boxer_Body_Mesh"
GUANTES = "Boxer_Gloves_Mesh"
SHORT = "Boxer_Shorts_Mesh"
OJOS = "Boxer_Eyes_Mesh"
RIG = "Boxer_Rig"

# Macros de MPFB2 en [0, 1]: varon, unos 25 anios, musculoso y magro, proporciones atleticas.
MACROS = {"gender": 1.0, "age": 0.5, "muscle": 1.0, "weight": 0.5, "proportions": 0.8, "height": 0.6}
# La piel LAYERED sale rosada y palida: se lleva a un tono trigueno multiplicando el color horneado.
TONO = (0.9, 0.74, 0.62)

# Guante de 14 oz, en metros: de la punta del punio al borde del punio de la manga.


def argumentos():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--salida", default="public/models/boxer_mpfb2.glb")
    p.add_argument("--textura", type=int, default=2048, help="lado de las texturas de piel")
    p.add_argument("--previa", default=None, help="png con un render de control (Cycles)")
    p.add_argument("--blend", default=None, help="guardar tambien el .blend para mirarlo")
    return p.parse_args(argv)


# -- utilidades ----------------------------------------------------------------------------

def activar(obj):
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def aplicar(obj, modificador):
    activar(obj)
    bpy.ops.object.modifier_apply(modifier=modificador.name)


def pos_hueso(rig, nombre, cola=False):
    b = rig.data.bones[nombre]
    return rig.matrix_world @ (b.tail_local if cola else b.head_local)


def pesos(obj, nombres):
    """Por vertice, la suma de los pesos en los grupos pedidos."""
    idx = {obj.vertex_groups[n].index for n in nombres if n in obj.vertex_groups}
    out = []
    for v in obj.data.vertices:
        out.append(sum(g.weight for g in v.groups if g.group in idx))
    return out


def borrar_vertices(obj, que):
    """Borra los vertices donde `que(i)` es verdadero."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if que(v.index)], context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def borrar_caras(obj, que):
    """Borra las caras donde `que(cara)` es verdadero, y los vertices que quedan sueltos."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if que(f)], context="FACES")
    sueltos = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=sueltos, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def triangulos(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def cycles(muestras=8):
    esc = bpy.context.scene
    esc.render.engine = "CYCLES"
    esc.cycles.samples = muestras
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for tipo in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = tipo
                prefs.get_devices()
                if any(d.type == tipo for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = d.type == tipo
                    esc.cycles.device = "GPU"
                    print(f"[avatar] Cycles en GPU ({tipo})")
                    return
            except TypeError:
                continue
    except Exception as e:  # sin GPU se hornea en CPU, mas lento
        print(f"[avatar] sin GPU para Cycles: {e}")
    esc.cycles.device = "CPU"


def imagen(nombre, lado, color=(0.5, 0.5, 0.5, 1.0), no_color=False):
    img = bpy.data.images.new(nombre, lado, lado, alpha=False)
    img.generated_color = color
    if no_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def hornear(obj, mat, img, tipo, **kw):
    """Hornea `tipo` de `mat` sobre `img`, con un nodo de imagen activo y suelto."""
    nodos = mat.node_tree.nodes
    n = nodos.new("ShaderNodeTexImage")
    n.image = img
    for x in nodos:
        x.select = False
    n.select = True
    nodos.active = n
    activar(obj)
    bpy.ops.object.bake(type=tipo, margin=8, **kw)
    nodos.remove(n)


def principled(nombre, base=None, color=(1, 1, 1, 1), rugosidad=None, rug_valor=0.5, normal=None, metal=0.0):
    """Material simple que el exportador glTF entiende entero."""
    m = bpy.data.materials.new(nombre)
    m.use_nodes = True
    t = m.node_tree
    bsdf = t.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Roughness"].default_value = rug_valor
    if base:
        n = t.nodes.new("ShaderNodeTexImage")
        n.image = base
        t.links.new(n.outputs["Color"], bsdf.inputs["Base Color"])
    if rugosidad:
        n = t.nodes.new("ShaderNodeTexImage")
        n.image = rugosidad
        t.links.new(n.outputs["Color"], bsdf.inputs["Roughness"])
    if normal:
        n = t.nodes.new("ShaderNodeTexImage")
        n.image = normal
        nm = t.nodes.new("ShaderNodeNormalMap")
        t.links.new(n.outputs["Color"], nm.inputs["Color"])
        t.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def material_relieve(nombre, escala, fuerza, rugosidad, detalle=8.0, voronoi=False):
    """Material temporal para hornear normal y rugosidad de un relieve procedural."""
    m = bpy.data.materials.new(nombre)
    m.use_nodes = True
    t = m.node_tree
    bsdf = t.nodes["Principled BSDF"]
    coord = t.nodes.new("ShaderNodeTexCoord")
    if voronoi:
        tex = t.nodes.new("ShaderNodeTexVoronoi")
        tex.feature = "SMOOTH_F1"
        tex.inputs["Scale"].default_value = escala
        salida = tex.outputs["Distance"]
    else:
        tex = t.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = escala
        tex.inputs["Detail"].default_value = detalle
        salida = tex.outputs["Fac"]
    t.links.new(coord.outputs["Object"], tex.inputs["Vector"])
    bump = t.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = fuerza
    t.links.new(salida, bump.inputs["Height"])
    t.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    # Rugosidad que varia con el mismo ruido: el sudor no brilla parejo.
    rampa = t.nodes.new("ShaderNodeMapRange")
    rampa.inputs["To Min"].default_value = rugosidad[0]
    rampa.inputs["To Max"].default_value = rugosidad[1]
    ruido = t.nodes.new("ShaderNodeTexNoise")
    ruido.inputs["Scale"].default_value = escala / 40
    t.links.new(coord.outputs["Object"], ruido.inputs["Vector"])
    t.links.new(ruido.outputs["Fac"], rampa.inputs["Value"])
    t.links.new(rampa.outputs["Result"], bsdf.inputs["Roughness"])
    return m


# -- 1. humano y rig ------------------------------------------------------------------------

def humano():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    macro = TargetService.get_default_macro_info_dict()
    macro.update(MACROS)
    cuerpo = HumanService.create_human(mask_helpers=False, detailed_helpers=True, extra_vertex_groups=True,
                                       feet_on_ground=True, scale=0.1, macro_detail_dict=macro)
    # Las macros quedan como shape keys: se fijan en la malla, si no el GLB lleva morphs.
    activar(cuerpo)
    if cuerpo.data.shape_keys:
        bpy.ops.object.shape_key_remove(all=True, apply_mix=True)
    rig = HumanService.add_builtin_rig(cuerpo, "game_engine", import_weights=True)
    cuerpo.name = CUERPO
    cuerpo.data.name = CUERPO
    rig.name = RIG
    return cuerpo, rig


def centro_grupo(obj, grupo):
    w = pesos(obj, [grupo])
    pts = [obj.matrix_world @ v.co for v, x in zip(obj.data.vertices, w) if x > 0.5]
    c = sum(pts, Vector()) / len(pts)
    r = sum((p - c).length for p in pts) / len(pts)
    return c, r


# -- 2. ojos -------------------------------------------------------------------------------

def ojos(cuerpo, rig):
    lado = 256
    img = bpy.data.images.new("ojo", lado, lado, alpha=False)
    px = [0.0] * (lado * lado * 4)
    for j in range(lado):
        for i in range(lado):
            d = math.hypot(i / lado - 0.5, j / lado - 0.5)
            if d < 0.075:
                c = (0.02, 0.015, 0.01)
            elif d < 0.17:
                k = (d - 0.075) / 0.095
                c = (0.18 + 0.12 * k, 0.1 + 0.06 * k, 0.04 + 0.02 * k)
            elif d < 0.19:
                c = (0.08, 0.05, 0.03)
            else:
                c = (0.86, 0.83, 0.8)
            o = (j * lado + i) * 4
            px[o:o + 4] = (*c, 1.0)
    img.pixels = px
    mat = principled("Ojos", base=img, rug_valor=0.15)

    partes = []
    for s, grupo in (("l", "helper-l-eye"), ("r", "helper-r-eye")):
        c, r = centro_grupo(cuerpo, grupo)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=14, radius=r * 0.97)
        uv = bm.loops.layers.uv.new("UVMap")
        # MPFB2 mira hacia -y: el iris se proyecta de frente sobre la mitad delantera.
        for f in bm.faces:
            for l in f.loops:
                p = l.vert.co / r
                l[uv].uv = (0.5 + 0.5 * p.x, 0.5 + 0.5 * p.z) if p.y < 0 else (0.02, 0.02)
        me = bpy.data.meshes.new(f"ojo_{s}")
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new(f"ojo_{s}", me)
        bpy.context.collection.objects.link(o)
        o.location = c
        partes.append(o)
    activar(partes[0])
    partes[1].select_set(True)
    bpy.ops.object.join()
    o = bpy.context.object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    o.name = OJOS
    o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    pegar_a_hueso(o, rig, lambda co: "head")
    return o


def pegar_a_hueso(obj, rig, hueso_de):
    """Pesa cada vertice entero a un hueso y lo deja deformado por el rig."""
    grupos = {}
    for v in obj.data.vertices:
        n = hueso_de(obj.matrix_world @ v.co)
        if n not in grupos:
            grupos[n] = obj.vertex_groups.new(name=n)
        grupos[n].add([v.index], 1.0, "REPLACE")
    obj.parent = rig
    obj.matrix_parent_inverse = rig.matrix_world.inverted()
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig


# -- 3. helpers y manos --------------------------------------------------------------------

DEDOS = ["thumb", "index", "middle", "ring", "pinky"]


def limpiar(cuerpo):
    cuerpo_w = pesos(cuerpo, ["body"])
    borrar_vertices(cuerpo, lambda i: cuerpo_w[i] < 0.5)
    for s in ("l", "r"):
        mano = [f"hand_{s}"] + [f"{d}_0{k}_{s}" for d in DEDOS for k in (1, 2, 3)]
        w = pesos(cuerpo, mano)
        borrar_vertices(cuerpo, lambda i: w[i] >= 0.5)
    for g in list(cuerpo.vertex_groups):
        if g.name.startswith(("helper-", "joint-")) or g.name in ("HelperGeometry", "JointCubes"):
            cuerpo.vertex_groups.remove(g)
    for m in list(cuerpo.modifiers):
        if m.type == "MASK":
            cuerpo.modifiers.remove(m)


# -- 4. short ------------------------------------------------------------------------------

def short(cuerpo, rig):
    H = cuerpo.dimensions.z
    cintura = pos_hueso(rig, "spine_01").z + 0.055 * H
    cadera = pos_hueso(rig, "thigh_l").z
    rodilla = pos_hueso(rig, "calf_l").z
    ruedo = rodilla + 0.5 * (cadera - rodilla)
    brazos = pesos(cuerpo, [f"{b}_{s}" for b in ("upperarm", "lowerarm", "hand") for s in "lr"])
    mw = cuerpo.matrix_world

    def adentro(f, margen=0.0):
        zs = [(mw @ v.co).z for v in f.verts]
        return all(ruedo + margen <= z <= cintura - margen for z in zs) and all(brazos[v.index] < 0.01 for v in f.verts)

    s = cuerpo.copy()
    s.data = cuerpo.data.copy()
    bpy.context.collection.objects.link(s)
    s.name = SHORT
    s.data.name = SHORT
    banda = 0.065 * H / 1.8
    # Bordes limpios: se corta con planos y no por caras enteras (quedaba un serrucho).
    borrar_caras(s, lambda f: any(brazos[v.index] >= 0.01 for v in f.verts))
    bm = bmesh.new()
    bm.from_mesh(s.data)
    todo = lambda: bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=todo(), dist=1e-5, plane_co=(0, 0, cintura), plane_no=(0, 0, 1), clear_outer=True)
    bmesh.ops.bisect_plane(bm, geom=todo(), dist=1e-5, plane_co=(0, 0, ruedo), plane_no=(0, 0, 1), clear_inner=True)
    bmesh.ops.bisect_plane(bm, geom=todo(), dist=1e-5, plane_co=(0, 0, cintura - banda), plane_no=(0, 0, 1))
    bm.to_mesh(s.data)
    bm.free()
    # La piel tapada no se ve y pincharia la tela al flexionar: se borra, menos un borde.
    borrar_caras(cuerpo, lambda f: adentro(f, margen=0.03))

    # Tela: afuera de la piel, mas holgada hacia el ruedo; la cintura ajustada y elastica.
    bm = bmesh.new()
    bm.from_mesh(s.data)
    bm.normal_update()
    for v in bm.verts:
        z = (mw @ v.co).z
        t = max(0.0, min(1.0, (cintura - banda - z) / max(cintura - banda - ruedo, 1e-6)))
        v.co += v.normal * (0.009 + 0.03 * t * t)
    for f in bm.faces:
        f.material_index = 1 if f.calc_center_median().z > cintura - banda else 0
    bm.to_mesh(s.data)
    bm.free()
    s.data.materials.clear()
    s.data.materials.append(principled("Short", color=(0.025, 0.025, 0.03, 1), rug_valor=0.55))
    s.data.materials.append(principled("Short_Cintura", color=(0.85, 0.85, 0.85, 1), rug_valor=0.4))
    sol = s.modifiers.new("Tela", "SOLIDIFY")
    sol.thickness = 0.005
    sol.offset = 1.0
    s.modifiers.move(len(s.modifiers) - 1, 0)
    aplicar(s, sol)
    activar(s)
    bpy.ops.object.shade_smooth()
    return s


# -- 5. guantes ----------------------------------------------------------------------------

def marco_mano(rig, s, cuerpo_x):
    """Ejes del guante: u a lo largo de la mano, v hacia el pulgar, w al dorso."""
    h0 = pos_hueso(rig, f"hand_{s}")
    u = (pos_hueso(rig, f"middle_01_{s}") - h0).normalized()
    t = pos_hueso(rig, f"thumb_01_{s}") - h0
    v = (t - u * t.dot(u)).normalized()
    w = u.cross(v)
    # Los dedos en reposo se curvan hacia la palma: el dorso queda del otro lado.
    punta = pos_hueso(rig, f"middle_03_{s}", cola=True) - h0
    curva = punta - u * punta.dot(u) - v * punta.dot(v)
    if curva.length > 1e-4:
        if curva.dot(w) > 0:
            w = -w
    # Sin curva medible, en la pose A de MPFB2 la palma mira al muslo.
    elif w.x * (h0.x - cuerpo_x) < 0:
        w = -w
    return h0, u, v, w


# Con rigidez 4 y umbral 0,6, la superficie de un elemento aislado queda a 0,684 de su radio
# (medido en Blender 4.2): asi cada pieza se escribe con su medida de superficie.
RIGIDEZ = 4.0
SUPERFICIE = 0.684


def elemento(mb, tipo, co, semiejes, direccion=None, largo=0.0, negativo=False):
    """Un elemento con su superficie en `semiejes` (metros), o un tubo de `largo` entre tapas."""
    r = max(semiejes)
    e = mb.elements.new(type=tipo)
    e.co = co
    e.radius = r / SUPERFICIE
    e.stiffness = RIGIDEZ
    e.use_negative = negativo
    if tipo == "CAPSULE":
        e.size_x, e.size_y, e.size_z = largo / 2, semiejes[1] / r, semiejes[2] / r
    else:
        e.size_x, e.size_y, e.size_z = (x / r for x in semiejes)
    if direccion is not None:
        e.rotation = Vector((1, 0, 0)).rotation_difference(Vector(direccion).normalized())
    return e


def forma_guante(nombre):
    """El guante en su marco local, en metros: x desde la munieca hacia los nudillos, y hacia
    el pulgar, z al dorso. Medidas de un 14 oz: 29 cm de largo con la manga, 13 de ancho."""
    mb = bpy.data.metaballs.new(nombre)
    mb.resolution = 0.004
    mb.threshold = 0.6
    E, C = "ELLIPSOID", "CAPSULE"
    # La cabeza del guante: el punio cerrado con el relleno, bastante mas grande que la manga.
    elemento(mb, E, (0.1, 0.0, 0.006), (0.078, 0.066, 0.06))
    # El domo de los nudillos, mas alto en el dorso: es lo que pega.
    elemento(mb, E, (0.125, 0.0, 0.022), (0.052, 0.06, 0.046))
    # La munieca se angosta: ahi el guante abraza la mano antes de la manga.
    elemento(mb, E, (0.025, 0.0, 0.0), (0.04, 0.046, 0.042))
    # Hueco de la palma donde se cierran los dedos: un elemento negativo apenas marcado.
    elemento(mb, E, (0.075, -0.004, -0.078), (0.045, 0.036, 0.022), negativo=True)
    # Pulgar cosido al costado, del lado de la palma, con la punta metida en la cabeza.
    elemento(mb, C, (0.075, 0.052, -0.026), (0.024, 0.024, 0.022), direccion=(1.0, -0.08, -0.1), largo=0.07)
    # Manga: tubo apenas eliptico hacia el antebrazo, con el borde un poco mas ancho.
    elemento(mb, C, (-0.05, 0.0, -0.002), (0.045, 0.045, 0.04), direccion=(1, 0, 0), largo=0.08)
    elemento(mb, E, (-0.095, 0.0, -0.002), (0.02, 0.049, 0.044))
    o = bpy.data.objects.new(nombre, mb)
    bpy.context.collection.objects.link(o)
    activar(o)
    bpy.ops.object.convert(target="MESH")
    return bpy.context.object


def medidas_guante(o):
    """Largo, ancho y alto del guante en su marco local; y donde empieza la manga."""
    xs, ys, zs = zip(*[v.co[:] for v in o.data.vertices])
    print(f"[avatar] guante {o.name}: largo {max(xs) - min(xs):.3f}, ancho {max(ys) - min(ys):.3f}, alto {max(zs) - min(zs):.3f} m")
    return min(xs)


def guante(rig, s, cuerpo_x, mat_cuero, mat_abrojo):
    o = forma_guante(f"guante_{s}")
    x_min = medidas_guante(o)
    # Remallado parejo y suavizado: los metaballs dejan triangulos desparejos.
    rem = o.modifiers.new("Remallar", "REMESH")
    rem.mode = "VOXEL"
    rem.voxel_size = 0.004
    aplicar(o, rem)
    lis = o.modifiers.new("Suavizar", "CORRECTIVE_SMOOTH")
    lis.factor = 0.8
    lis.iterations = 8
    lis.smooth_type = "LENGTH_WEIGHTED"
    lis.use_only_smooth = True
    aplicar(o, lis)
    # Despues de remallar (el voxel pide malla cerrada), la manga queda abierta: por ahi entra el antebrazo.
    bm = bmesh.new()
    bm.from_mesh(o.data)
    bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-5,
                           plane_co=(x_min + 0.012, 0, 0), plane_no=(1, 0, 0), clear_inner=True)
    bm.to_mesh(o.data)
    bm.free()
    x_min += 0.012
    # Abrojo: una banda de 4 cm sobre la manga, apenas en relieve, con la costura del borde.
    bm = bmesh.new()
    bm.from_mesh(o.data)
    lo, hi = x_min + 0.035, x_min + 0.085
    for v in bm.verts:
        if lo <= v.co.x <= hi:
            radial = Vector((0, v.co.y, v.co.z))
            if radial.length > 1e-6:
                v.co += radial.normalized() * 0.003
    for f in bm.faces:
        cx = f.calc_center_median().x
        f.material_index = 1 if lo <= cx <= hi else 0
    bm.to_mesh(o.data)
    bm.free()
    dec = o.modifiers.new("Decimar", "DECIMATE")
    dec.ratio = min(1.0, 2600 / max(len(o.data.polygons), 1))
    aplicar(o, dec)
    o.data.materials.append(mat_cuero)
    o.data.materials.append(mat_abrojo)
    activar(o)
    bpy.ops.object.shade_smooth()
    # Del marco local al mundo, sobre la mano del rig.
    h0, u, v, w = marco_mano(rig, s, cuerpo_x)
    o.matrix_world = Matrix((
        (u.x, v.x, w.x, h0.x),
        (u.y, v.y, w.y, h0.y),
        (u.z, v.z, w.z, h0.z),
        (0, 0, 0, 1),
    ))
    # El guante izquierdo es el espejo del derecho: v se dio vuelta y la malla quedaria al reves.
    activar(o)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if o.matrix_world.determinant() < 0 or (u.cross(v)).dot(w) < 0:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
    return o


def guantes(cuerpo, rig, lado_tex):
    cuerpo_x = (cuerpo.matrix_world @ Vector()).x
    # Cuero: grano fino y rugosidad media; se hornea normal y rugosidad sobre una UV propia.
    tmp = material_relieve("cuero_tmp", escala=900.0, fuerza=0.25, rugosidad=(0.28, 0.45), voronoi=True)
    partes = []
    for s in ("l", "r"):
        partes.append(guante(rig, s, cuerpo_x, tmp, tmp))
    activar(partes[0])
    partes[1].select_set(True)
    bpy.ops.object.join()
    o = bpy.context.object
    o.name = GUANTES
    o.data.name = GUANTES
    activar(o)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")

    lado = max(512, lado_tex // 2)
    nrm = imagen("guante_normal", lado, (0.5, 0.5, 1.0, 1.0), no_color=True)
    rug = imagen("guante_rugosidad", lado, no_color=True)
    hornear(o, tmp, nrm, "NORMAL", normal_space="TANGENT")
    hornear(o, tmp, rug, "ROUGHNESS")
    o.data.materials.clear()
    # Base blanca: la web la tine con el color del lado (A o B). El abrojo queda fijo.
    o.data.materials.append(principled("Guante_Cuero", color=(1, 1, 1, 1), rugosidad=rug, normal=nrm))
    o.data.materials.append(principled("Guante_Detalle", color=(0.92, 0.92, 0.9, 1), rug_valor=0.7, normal=nrm))
    bpy.data.materials.remove(tmp)

    def mano(co):
        return "hand_l" if (co - pos_hueso(rig, "hand_l")).length < (co - pos_hueso(rig, "hand_r")).length else "hand_r"
    pegar_a_hueso(o, rig, mano)
    return o


# -- 6. piel -------------------------------------------------------------------------------

def piel(cuerpo, lado):
    """Hornea la piel LAYERED de MPFB2 a color y rugosidad, y una normal de poros."""
    HumanService.set_character_skin(None, cuerpo, skin_type="LAYERED")
    capas = cuerpo.data.materials[0]
    color = imagen("piel_color", lado)
    rug = imagen("piel_rugosidad", lado, no_color=True)
    nrm = imagen("piel_normal", lado, (0.5, 0.5, 1.0, 1.0), no_color=True)
    hornear(cuerpo, capas, color, "DIFFUSE", pass_filter={"COLOR"})
    px = list(color.pixels)
    for i in range(0, len(px), 4):
        px[i], px[i + 1], px[i + 2] = px[i] * TONO[0], px[i + 1] * TONO[1], px[i + 2] * TONO[2]
    color.pixels = px
    # Poros y sudor: rugosidad baja y despareja, que brille donde le pega la luz.
    tmp = material_relieve("piel_tmp", escala=1400.0, fuerza=0.08, rugosidad=(0.3, 0.55), detalle=10.0)
    cuerpo.data.materials[0] = tmp
    hornear(cuerpo, tmp, nrm, "NORMAL", normal_space="TANGENT")
    hornear(cuerpo, tmp, rug, "ROUGHNESS")
    cuerpo.data.materials[0] = principled("Piel", base=color, rugosidad=rug, normal=nrm)
    bpy.data.materials.remove(tmp)
    activar(cuerpo)
    bpy.ops.object.shade_smooth()


# -- 7. export y previa --------------------------------------------------------------------

def exportar(salida, objetos, rig):
    os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in [rig, *objetos]:
        o.select_set(True)
    bpy.context.view_layer.objects.active = rig
    rig.data.pose_position = "REST"
    deseadas = dict(
        filepath=os.path.abspath(salida), export_format="GLB", use_selection=True, export_apply=True,
        export_animations=False, export_skins=True, export_morph=False, export_yup=True,
        export_image_format="JPEG", export_jpeg_quality=88, export_image_quality=88,
        export_texcoords=True, export_normals=True, export_tangents=False, export_cameras=False,
        export_lights=False, export_extras=False,
    )
    validas = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
    bpy.ops.export_scene.gltf(**{k: v for k, v in deseadas.items() if k in validas})
    print(f"[avatar] exportado {salida} ({os.path.getsize(salida) / 2**20:.2f} MB)")


def previa(ruta, cuerpo):
    esc = bpy.context.scene
    H = cuerpo.dimensions.z
    cam = bpy.data.objects.new("camara", bpy.data.cameras.new("camara"))
    bpy.context.collection.objects.link(cam)
    cam.location = (1.4, -3.6, 1.15)
    cam.rotation_euler = (math.radians(86), 0, math.radians(21))
    cam.data.lens = 50
    esc.camera = cam
    for nombre, energia, lugar in (("clave", 900, (2.5, -3, 3.5)), ("contra", 700, (-2.5, 3, 2.5)), ("relleno", 250, (-3, -2, 1.5))):
        luz = bpy.data.objects.new(nombre, bpy.data.lights.new(nombre, "AREA"))
        luz.data.energy = energia
        luz.data.size = 1.5
        luz.location = lugar
        luz.rotation_euler = (Vector((0, 0, 0.5 * H)) - Vector(lugar)).to_track_quat("-Z", "Y").to_euler()
        bpy.context.collection.objects.link(luz)
    if esc.world is None:
        esc.world = bpy.data.worlds.new("mundo")
    esc.world.color = (0.02, 0.022, 0.026)
    esc.render.resolution_x, esc.render.resolution_y = 720, 1080
    esc.cycles.samples = 64
    esc.render.filepath = os.path.abspath(ruta)
    bpy.ops.render.render(write_still=True)


def main():
    a = argumentos()
    cycles()
    cuerpo, rig = humano()
    o_ojos = ojos(cuerpo, rig)
    limpiar(cuerpo)
    o_short = short(cuerpo, rig)
    o_guantes = guantes(cuerpo, rig, a.textura)
    piel(cuerpo, a.textura)
    for o in (cuerpo, o_short, o_guantes, o_ojos):
        print(f"[avatar] {o.name}: {triangulos(o)} triangulos")
    print(f"[avatar] total {sum(triangulos(o) for o in (cuerpo, o_short, o_guantes, o_ojos))} triangulos")
    if a.blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.blend))
    exportar(a.salida, [cuerpo, o_short, o_guantes, o_ojos], rig)
    if a.previa:
        previa(a.previa, cuerpo)


if __name__ == "__main__":
    main()
