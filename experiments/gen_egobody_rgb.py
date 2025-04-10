import smplx
import torch
import random
import pickle
import trimesh
import tqdm
import pyrender
import numpy as np
import glob
import subprocess
import cv2
import copy
from PIL import Image
import pdb
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from scipy.spatial.transform import Rotation
import pyquaternion as pyquat
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx
from OpenGL.GL import *
jet = plt.get_cmap('twilight')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

def create_simple_cubemap(image_path):
    # Load one image for all faces (test purposes only)
    image = Image.open(image_path).convert("RGB").resize((512, 512))
    img_data = np.array(image).astype(np.uint8)

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex_id)
    for i in range(6):
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    return tex_id
def render_cubemap(scene, env_camera, renderer, resolution=256):
    """
    Renders the six faces of a cubemap from the scene.
    
    Parameters:
        scene (pyrender.Scene): The scene to capture.
        env_camera (pyrender.Camera): The camera used for cubemap rendering.
        renderer (pyrender.OffscreenRenderer): Renderer instance.
        resolution (int): Resolution (width & height) for each cubemap face.
    
    Returns:
        dict: A dictionary with keys for each face ('posx', 'negx', 'posy', 'negy', 'posz', 'negz')
              and their corresponding rendered images (numpy arrays).
    """
    # Define the look directions and up vectors for each cubemap face.
    faces = {
        'posx': (np.array([1, 0, 0]),  np.array([0, -1, 0])),
        'negx': (np.array([-1, 0, 0]), np.array([0, -1, 0])),
        'posy': (np.array([0, 1, 0]),  np.array([0, 0, 1])),
        'negy': (np.array([0, -1, 0]), np.array([0, 0, -1])),
        'posz': (np.array([0, 0, 1]),  np.array([0, -1, 0])),
        'negz': (np.array([0, 0, -1]), np.array([0, -1, 0])),
    }
    
    cubemap = {}
    # Save the original camera pose if needed.
    original_camera_pose = np.eye(4)
    
    # For each face, set up a new camera node with the desired orientation.
    for face, (look_dir, up_dir) in faces.items():
        cam_pose = np.eye(4)
        # Set forward direction (-z in camera space) to be opposite of the look_dir.
        cam_pose[:3, 2] = -look_dir
        cam_pose[:3, 1] = up_dir
        cam_pose[:3, 0] = np.cross(up_dir, -look_dir)
        # Position the camera at the scene's center.
        cam_pose[:3, 3] = original_camera_pose[:3, 3]
        
        # Remove any existing environment camera node.
        # for node in scene.get_nodes(camera=env_camera):
        #     scene.remove_node(node)
        cam_node = pyrender.Node(camera=env_camera, matrix=cam_pose)
        scene.add_node(cam_node)
        
        # Render this face.
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        cubemap[face] = color
        # Remove the temporary camera node.
        scene.remove_node(cam_node)
    
    return cubemap

def sample_face(face_img, u, v):
    """
    Bilinearly sample from face_img at texture coordinates u,v.
    face_img is assumed to be a numpy array of shape (H, H, C) in uint8.
    u and v should be numpy arrays with values in [0, 1].
    Returns a numpy array of shape (N, C) where N is the number of sample points.
    """
    H = face_img.shape[0]
    # Convert normalized coordinates to pixel space
    x = u * (H - 1)
    y = v * (H - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)

    # Compute interpolation weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # Sample four corners (note: image indexing is [row, col])
    Ia = face_img[y0, x0, :3].astype(np.float32)
    Ib = face_img[y1, x0, :3].astype(np.float32)
    Ic = face_img[y0, x1, :3].astype(np.float32)
    Id = face_img[y1, x1, :3].astype(np.float32)

    # Combine with weights
    result = (wa[..., None] * Ia + wb[..., None] * Ib +
              wc[..., None] * Ic + wd[..., None] * Id)
    return result

def cubemap_to_equirectangular(cubemap, width=1024, height=512):
    """
    Converts a cubemap (dictionary of 6 face images) into an equirectangular image.
    The cubemap dictionary must contain keys: 'posx', 'negx', 'posy', 'negy', 'posz', 'negz'.
    Each face image should be a numpy array (e.g. rendered via pyrender) of shape (R, R, 4) or (R, R, 3).

    Returns:
        A PIL.Image instance containing the stitched equirectangular environment map.
    """
    # Create output equirectangular image array.
    eq_img = np.zeros((height, width, 3), dtype=np.float32)

    # Create a grid of pixel coordinates in the equirectangular image.
    j, i = np.meshgrid(np.arange(width), np.arange(height))
    # Normalize to [0,1]
    u = i.astype(np.float32) / (height - 1)
    v = j.astype(np.float32) / (width - 1)

    # Convert to spherical coordinates.
    # theta: longitude in [-pi, pi], phi: latitude in [0, pi]
    theta = (u * np.pi * 2) - np.pi
    phi = v * np.pi

    # Convert spherical coordinates to Cartesian directions.
    # Here: x = sin(phi)*cos(theta), y = cos(phi), z = sin(phi)*sin(theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.cos(phi)
    z = np.sin(phi) * np.sin(theta)

    abs_x = np.abs(x)
    abs_y = np.abs(y)
    abs_z = np.abs(z)

    # Determine which face of the cubemap each direction hits.
    face_idx = np.empty_like(x, dtype=np.int32)
    # 0: posx, 1: negx, 2: posy, 3: negy, 4: posz, 5: negz
    # For each pixel, compare absolute components.
    is_x_dominant = (abs_x >= abs_y) & (abs_x >= abs_z)
    is_y_dominant = (abs_y >= abs_x) & (abs_y >= abs_z)
    is_z_dominant = (abs_z >= abs_x) & (abs_z >= abs_y)

    face_idx[is_x_dominant & (x > 0)] = 0  # posx
    face_idx[is_x_dominant & (x <= 0)] = 1  # negx
    face_idx[is_y_dominant & (y > 0)] = 2  # posy
    face_idx[is_y_dominant & (y <= 0)] = 3  # negy
    face_idx[is_z_dominant & (z > 0)] = 4  # posz
    face_idx[is_z_dominant & (z <= 0)] = 5  # negz

    # Mapping from face index to cubemap key.
    code_to_key = {0: 'posx', 1: 'negx', 2: 'posy',
                   3: 'negy', 4: 'posz', 5: 'negz'}

    # Prepare arrays for texture coordinates for each pixel.
    s = np.zeros_like(x, dtype=np.float32)
    t = np.zeros_like(x, dtype=np.float32)

    # Compute texture coordinates for each face using standard formulas.
    # For posx (face 0)
    mask = (face_idx == 0)
    s[mask] = (-z[mask] / abs_x[mask] + 1) / 2
    t[mask] = (-y[mask] / abs_x[mask] + 1) / 2

    # For negx (face 1)
    mask = (face_idx == 1)
    s[mask] = (z[mask] / abs_x[mask] + 1) / 2
    t[mask] = (-y[mask] / abs_x[mask] + 1) / 2

    # For posy (face 2)
    mask = (face_idx == 2)
    s[mask] = (x[mask] / abs_y[mask] + 1) / 2
    t[mask] = (-z[mask] / abs_y[mask] + 1) / 2

    # For negy (face 3)
    mask = (face_idx == 3)
    s[mask] = (x[mask] / abs_y[mask] + 1) / 2
    t[mask] = (z[mask] / abs_y[mask] + 1) / 2

    # For posz (face 4)
    mask = (face_idx == 4)
    s[mask] = (x[mask] / abs_z[mask] + 1) / 2
    t[mask] = (-y[mask] / abs_z[mask] + 1) / 2

    # For negz (face 5)
    mask = (face_idx == 5)
    s[mask] = (-x[mask] / abs_z[mask] + 1) / 2
    t[mask] = (-y[mask] / abs_z[mask] + 1) / 2

    # For each face, sample from the corresponding cubemap face.
    for face_code in range(6):
        key = code_to_key[face_code]
        face_img = cubemap[key]
        # Ensure face image is in uint8 and in RGB (drop alpha if present)
        if face_img.shape[2] == 4:
            face_img = face_img[:, :, :3]
        # Get indices in the equirectangular image that map to this face.
        face_mask = (face_idx == face_code)
        if np.any(face_mask):
            # Extract the texture coordinates for these pixels.
            s_face = s[face_mask]
            t_face = t[face_mask]
            # Sample color values using bilinear interpolation.
            sampled = sample_face(face_img, s_face, t_face)
            eq_img[face_mask] = sampled

    # Convert result to uint8 and create a PIL Image.
    eq_img = np.clip(eq_img, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(eq_img)
    # pil_img.save('cubemap_equirectangular.png')
    return Image.fromarray(eq_img)

def add_skybox(scene, env_texture):
    """
    Creates a skybox (a large inverted sphere) textured with the provided environment map.
    
    Parameters:
        scene (pyrender.Scene): The scene to add the skybox to.
        env_texture (PIL.Image): The environment map as an image.
    """
    # Create a large sphere to act as a skybox.
    sky_sphere = trimesh.creation.icosphere(radius=100, subdivisions=4)
    sky_sphere.invert()  # Invert normals so the texture is visible from inside.
    
    # Create a PBR material that uses the environment texture.
    sky_material = pyrender.MetallicRoughnessMaterial(
        baseColorTexture=pyrender.Texture(source=env_texture, source_channels='RGBA'),
        metallicFactor=1.0,
        roughnessFactor=0.0
    )
    
    skybox_mesh = pyrender.Mesh.from_trimesh(sky_sphere, material=sky_material)
    skybox_node = pyrender.Node(mesh=skybox_mesh, name='skybox')
    scene.add_node(skybox_node)

def create_pbr_mesh(trimesh_obj, texture_image, roughness=0.5):
    """
    Create a mesh with a PBR material that can reflect an environment map.
    
    Parameters:
        trimesh_obj (trimesh.Trimesh): The mesh to be rendered.
        texture_image (PIL.Image): The texture image for the object.
        roughness (float): Roughness factor for reflections.
    
    Returns:
        pyrender.Mesh: Mesh with a PBR material.
    """
    # pbr_material = pyrender.MetallicRoughnessMaterial(
    #     baseColorTexture=pyrender.Texture(source=texture_image, source_channels='RGBA'),
    #     metallicFactor=0.0,
    #     roughnessFactor=roughness
    # )
    pbr_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0, 0, 0, 0],
        metallicFactor=0.0,
        roughnessFactor=roughness
    )
    return pyrender.Mesh.from_trimesh(trimesh_obj, material=pbr_material, smooth=True)

def add_mirror_sphere(scene, position=(0, 1, 0), radius=0.5):
    """
    Adds a highly reflective mirror sphere to the scene to visualize cubemap reflections.
    
    Parameters:
        scene (pyrender.Scene): The pyrender scene to add the sphere to.
        position (tuple): (x, y, z) position of the sphere.
        radius (float): Radius of the sphere.
    """
    # Create a simple sphere mesh using trimesh
    sphere_mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)

    # Create a PBR material with metallic and low roughness (mirror-like)
    mirror_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],  # white so reflections are clean
        metallicFactor=1.0,                   # full metal
        roughnessFactor=0.0                   # perfect mirror
    )

    # Create pyrender mesh
    mesh = pyrender.Mesh.from_trimesh(sphere_mesh, material=mirror_material, smooth=True)

    # Create a node at the desired position
    node = pyrender.Node(mesh=mesh, translation=np.array(position))

    # Add to the scene
    scene.add_node(node)

def make_new_mesh(vt, f, ft, mesh, image):
    """
    Add missing vertices to the mesh such that it has the same number of vertices as the texture coordinates
    mesh: 3D vertices of the orginal mesh
    vt: 2D vertices of the texture map
    f: 3D faces of the orginal mesh (0-indexed)
    ft: 2D faces of the texture map (0-indexed)
    """
    #build a correspondance dictionary from the original mesh indices to the (possibly multiple) texture map indices
    f_flat = f.flatten()
    ft_flat = ft.flatten()
    correspondances = {}

    #traverse and find the corresponding indices in f and ft
    for i in range(len(f_flat)):
        if f_flat[i] not in correspondances:
            correspondances[f_flat[i]] = [ft_flat[i]]
        else:
            if ft_flat[i] not in correspondances[f_flat[i]]:
                correspondances[f_flat[i]].append(ft_flat[i])

    #build a mesh using the texture map vertices
    new_mesh = np.zeros((vt.shape[0], 3))
    for old_index, new_indices in correspondances.items():
        for new_index in new_indices:
            new_mesh[new_index] = mesh[old_index]

    return trimesh.Trimesh(vertices=new_mesh, faces=ft, \
                            visual=trimesh.visual.TextureVisuals(uv=vt, image=image), process=False)


def obj_vt(fname): # read texture coordinates: (u,v)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('vt '):
                tmp = line.split(' ')
                v = [float(i) for i in tmp[1:3]]
                res.append(v)
    return np.array(res, dtype=np.float32)

def obj_fv(fname): # read vertices id in faces: (vv1,vv2,vv3)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('f '):
                tmp = line.split(' ')
                if '/' in tmp[1]:
                    v = [int(i.split('/')[0]) for i in tmp[1:4]]
                else:
                    v = [int(i) for i in tmp[1:4]]
                res.append(v)
    return np.array(res, dtype=np.int32) - 1 # obj index from 1

def obj_ft(fname): # read texture id in faces: (vt1,vt2,vt3)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('f '):
                tmp = line.split(' ')
                if '/' in tmp[1]:
                    v = [int(i.split('/')[1]) for i in tmp[1:4]]
                else:
                    raise(Exception("not a textured obj file"))
                res.append(v)
    return np.array(res, dtype=np.int32) - 1 # obj index from 1

def adjust_global_orient(global_orient):
    theta = -np.pi / 2
    sin = np.sin(theta)
    cos = np.cos(theta)

    rot = Rotation.from_rotvec(global_orient)
    global_orient_matrix = rot.as_matrix()

    Rot = np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])  # Rx

    global_orient_matrix = Rot @ global_orient_matrix
    rot = Rotation.from_matrix(global_orient_matrix)
    global_orient = rot.as_rotvec()

    return global_orient


def adjust_transl(transl):
    transl = transl[:, [0, 2, 1]]
    transl[:, 2] *= -1
    return transl

def adjust_back(verts, R0, T0, R, T, pelvis):
    R0 = Rotation.from_rotvec(R0).as_matrix()
    R = Rotation.from_rotvec(R).as_matrix()
    T = T[:, np.newaxis, :]
    T0 = T0[:, np.newaxis, :]
    return np.einsum('bij,btj->bti', R0 @ R.transpose(0, 2, 1), \
                                    verts - pelvis[:, np.newaxis, :] - T)\
           + pelvis[:, np.newaxis, :] + T0

def params2torch(params, dtype=torch.float32):
    return {k: torch.cuda.FloatTensor(v) if type(v) == np.ndarray else v for k, v in params.items()}

def rollout_primitives(motion_primitives):
    smplx_param_list = []
    gender = motion_primitives[0]['gender']

    if gender == 'male':
        body_model = bm_male20
    elif gender == 'female':
        body_model = bm_female20
    else:
        body_model = None
        pdb.set_trace()

    for idx, motion_primitive in enumerate(motion_primitives):
        pelvis_original = body_model(betas=torch.cuda.FloatTensor(motion_primitive['betas']).repeat(20, 1)).joints[:, 0, :].detach().cpu().numpy()  # [20, 3]
        smplx_param = motion_primitive['smplx_params'][0]  #[10, 96]

        rotation = motion_primitive['transf_rotmat'].reshape((3, 3)) # [3, 3]
        transl = motion_primitive['transf_transl'].reshape((1, 3)) # [1, 3]
        smplx_param[:, :3] = np.matmul((smplx_param[:, :3] + pelvis_original), rotation.T) - pelvis_original + transl
        r_ori = Rotation.from_rotvec(smplx_param[:, 3:6])
        r_new = Rotation.from_matrix(np.tile(motion_primitive['transf_rotmat'], [20, 1, 1])) * r_ori
        smplx_param[:, 3:6] = r_new.as_rotvec()

        if idx == 0:
            start_frame = 0
        elif motion_primitive['mp_type'] == '1-frame':
            start_frame = 1
        elif motion_primitive['mp_type'] == '2-frame':
            start_frame = 2
        else:
            # crowd-env use 1 frame model at the moment
            start_frame = 1
        smplx_param = smplx_param[start_frame:, :]
        smplx_param_list.append(smplx_param)

    return  np.concatenate(smplx_param_list, axis=0)  # [t, 96]


def gen_data_egobody(vis_marker=False, vis_pelvis=True, vis_object=False,
                vis_navmesh=True, start_frame=0,
                slow_rate=1, save_path=None, add_floor=True, scene_mesh=None, scene_name=None):
    ambient_intensity = np.random.uniform(0.5, 0.8)
    bg_color = np.random.uniform(0.9, 1.)
    scene = pyrender.Scene(ambient_light=[ambient_intensity, ambient_intensity, ambient_intensity], bg_color=[bg_color, bg_color, bg_color, 0.5])
    motions_list = []

    m = pyrender.Mesh.from_trimesh(scene_mesh)
    object_node = pyrender.Node(mesh=m, name='scene')
    scene.add_node(object_node)

    # eval model
    # while True:
    #     # ret = subprocess.call(['python', 'crowd_ppo/main_egobody_eval.py', '--resume-path=/mnt/vlg-nfs/genli/log_pretrain_dep13_seedori/log_2f_ego_gru_rpene1_rlook0.3-finetune-newrpene0.1/collision-avoidance/ppo/0/231017-222547/policy.pth', '--watch', '--scene-name=%s' % scene_name])
    #     ret = subprocess.call(['python', 'crowd_ppo/main_egobody_eval.py', '--resume-path=data/checkpoint_best.pth', '--watch', '--scene-name=%s' % scene_name])
    #     if ret == 0:
    #         break
    result_paths = ['egobody_tmp_res/motion_0.pkl', 'egobody_tmp_res/motion_1.pkl']
    for input_path in result_paths:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            motions = data['motion']
            motions_list.append(motions)

    rollout_frames_list = [rollout_primitives(motions) for motions in motions_list]
    print(np.array([len(frames) for frames in rollout_frames_list]))
    max_frame = np.array([len(frames) for frames in rollout_frames_list]).max()

    # rollout_frames_pad_list = []  # [T_max, 93], pad shorter sequences with last frame
    # for idx in range(len(rollout_frames_list)):
    #     frames = rollout_frames_list[idx]
    #     rollout_frames_pad_list.append(np.concatenate([frames, np.tile(frames[-1:, :], (max_frame + 1 - frames.shape[0], 1))], axis=0))
    # smplx_params = np.stack(rollout_frames_pad_list, axis=0)  # [S, T_max, 93]
    smplx_params = np.stack(rollout_frames_list, axis=0)
    betas = [motions[0]['betas'] for motions in motions_list]
    betas = np.stack(betas, axis=0)  # [S, 10]
    genders = [motions[0]['gender'] for motions in motions_list]
    genders = np.stack(genders, axis=0)
    if genders[0] != genders[1]:
        pdb.set_trace()

    smpl_texture_path = None
    if genders[0] == 'male':
        body_model = bm_male2 
        ethnicity = random.choice(["asian", "hispanic", "mideast", "white"])
        texture_paths = [tp for tp in body_texture_path if "m_" + ethnicity in tp]
        smpl_texture_path = random.choice(texture_paths).split('/')[-1] 
        # clothing_name = random.choice(['rp_aaron_posed_009', "rp_aaron_posed_013", "rp_ethan_posed_015", "rp_henry_posed_001"])
        clothing_name = random.choice(['rp_aaron_posed_009', "rp_aaron_posed_013"]) 
        # clothing_texture_name = random.choice(clothing_texture_paths).split('/')[-2]
    elif genders[0] == 'female':
        body_model = bm_female2
        ethnicity = random.choice(["asian", "hispanic", "mideast", "white"])
        texture_paths = [tp for tp in body_texture_path if "f_" + ethnicity in tp]
        smpl_texture_path = random.choice(texture_paths).split('/')[-1]
        # clothing_name = random.choice(["rp_alexandra_posed_025", "rp_aneko_posed_011", "rp_claudia_posed_020"])
        clothing_name = random.choice(["rp_alexandra_posed_025", "rp_aneko_posed_011"]) 
        # clothing_texture_name = random.choice(clothing_texture_paths).split('/')[-2]
    else:
        body_model = None
        pdb.set_trace()
    clothing_texture_names = [x for x in clothing_texture_paths if clothing_name in x]
    clothing_texture_name = random.choice(clothing_texture_names).split('/')[-2]
    print(clothing_name, clothing_texture_name)

    # top_uv = trimesh.load('/mnt/vlg-nfs/genli/datasets/bedlam/clothing_meshes/%s/top.obj' % (clothing_name), process=False) 
    # top_uv.merge_vertices(merge_tex=True)
    # top_uv = top_uv.visual.uv
    top_vt = obj_vt('HOOD/hood_data/bedlam/clothing_meshes/%s/top.obj' % (clothing_name))
    top_f = obj_fv('HOOD/hood_data/bedlam/clothing_meshes/%s/top.obj' % (clothing_name))
    top_ft = obj_ft('HOOD/hood_data/bedlam/clothing_meshes/%s/top.obj' % (clothing_name))

    # pant_uv = trimesh.load('/mnt/vlg-nfs/genli/datasets/bedlam/clothing_meshes/%s/pant.obj' % (clothing_name), process=False)
    # pant_uv.merge_vertices(merge_tex=True)
    # pant_uv = pant_uv.visual.uv
    pant_vt = obj_vt('HOOD/hood_data/bedlam/clothing_meshes/%s/pant.obj' % (clothing_name))
    pant_f = obj_fv('HOOD/hood_data/bedlam/clothing_meshes/%s/pant.obj' % (clothing_name))
    pant_ft = obj_ft('HOOD/hood_data/bedlam/clothing_meshes/%s/pant.obj' % (clothing_name))

    body_node = None
    camera_node = None
    top_node = None
    pant_node = None

    cx = np.random.uniform(942.543, 946.108)
    cy = np.random.uniform(505.898, 510.081)
    fx = np.random.uniform(1450.93, 1480.28)

    renderer = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080)
    
    # Init camera here
    camera_pose = np.eye(4)
    camera = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fx, cx=cx, cy=cy)
    light = pyrender.DirectionalLight(color=[np.random.uniform(0.9, 1.), np.random.uniform(0.9, 1.), np.random.uniform(0.9, 1.)],
                                      intensity=np.random.uniform(2., 6.))
    # light = pyrender.SpotLight(color=np.ones(3), intensity=6.)
    light_node = pyrender.Node(light=light)
    scene.add_node(light_node)

    # Create an environment camera for cubemap capture.
    # env_camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    # Render cubemap faces from the center of the scene.
    # cubemap = render_cubemap(scene, env_camera, renderer, resolution=256)
    # Convert the cubemap to an equirectangular environment map.
    # env_map = cubemap_to_equirectangular(cubemap, width=1024, height=512)
    env_cubemap_id = create_simple_cubemap("cubemap_equirectangular.png")
    renderer.env_cubemap = env_cubemap_id
    # env_map = Image.open('synthetic_env_map.png')
    # env_map = env_map.convert("RGBA")

    # Add a skybox to the scene that uses the environment map.
    # add_skybox(scene, env_map)
    # add_mirror_sphere(scene, position=(0, 1, 0), radius=0.4)

    top = {0: {}, 1: {}}
    pant = {0: {}, 1: {}}
    for man_id in range(2):
        # simulate cloth with hood
        hood = {
            "betas": betas[man_id],
            "gender": genders[man_id], 
            "transl": smplx_params[man_id, :, :3],
            "global_orient": smplx_params[man_id, :, 3:6],
            "body_pose": smplx_params[man_id, :, 6:69]
            }
        ori_glob_orient = copy.deepcopy(hood['global_orient'])
        ori_transl = copy.deepcopy(hood['transl'])
        hood['global_orient'] = adjust_global_orient(hood['global_orient'])
        hood['transl'] = adjust_transl(hood['transl'])
        with open("./pose_seq_tmp.pkl", 'wb') as f:
            pickle.dump(hood, f)

        # call hood eval to get clothing verts
        curpath = os.getcwd()
        os.chdir(os.path.join(os.getcwd(), 'HOOD'))

        print('Simulating garment %d top ...' % man_id)
        subprocess.call([HOOD_PYTHON, 'eval.py', '--seq-name', os.path.join(curpath, 'pose_seq_tmp.pkl'), \
                                                 '--garment-name', clothing_name + '_top', \
                                                 '--bz', str(hood['body_pose'].shape[0]),
                                                 '--gender', genders[man_id]])

        print('Simulating garment %d pant ...' % man_id)
        subprocess.call([HOOD_PYTHON, 'eval.py', '--seq-name', os.path.join(curpath, 'pose_seq_tmp.pkl'), \
                                                 '--garment-name', clothing_name + '_pant', \
                                                 '--bz', str(hood['body_pose'].shape[0]),
                                                 '--gender', genders[man_id]])
        os.chdir(curpath)
        top_data = np.load('HOOD/hood_data/temp/output_%s_top.pkl' % clothing_name, allow_pickle=True)
        top_verts = top_data['pred']
        top_faces = top_data['cloth_faces']
        pant_data = np.load('HOOD/hood_data/temp/output_%s_pant.pkl' % clothing_name, allow_pickle=True)
        pant_verts = pant_data['pred']
        pant_faces = pant_data['cloth_faces']

        # adjust axis coord back to egogen
        # hood discard first 2 frames
        pelvis = body_model(betas=torch.cuda.FloatTensor(hood['betas']).unsqueeze(0)).joints[:1, 0, :].detach().cpu().numpy()
        top_verts = adjust_back(top_verts, ori_glob_orient[2:], ori_transl[2:],\
                                           hood['global_orient'][2:], hood['transl'][2:], pelvis)
        pant_verts = adjust_back(pant_verts, ori_glob_orient[2:], ori_transl[2:],\
                                           hood['global_orient'][2:], hood['transl'][2:], pelvis)

        top[man_id]['verts'] = top_verts
        top[man_id]['faces'] = top_faces
        pant[man_id]['verts'] = pant_verts
        pant[man_id]['faces'] = pant_faces

    for frame_idx in tqdm.tqdm(range(start_frame, max_frame)):
        if frame_idx < 2: # dicard first 2 frames because of HOOD 
            continue
        # keep frames when social distance between 1m-5m
        flag_dist = False
        smplx_transl = smplx_params[:, frame_idx, :3]
        smplx_transl_dist = np.linalg.norm(smplx_transl[0] - smplx_transl[1])
        if smplx_transl_dist >= 1 and smplx_transl_dist <= 5:
            flag_dist = True
        else:
            continue

        smplx_dict = {
            'betas': betas,
            'transl': smplx_params[:, frame_idx, :3],
            'global_orient': smplx_params[:, frame_idx, 3:6],
            'body_pose': smplx_params[:, frame_idx, 6:69],
        }
        smplx_dict = params2torch(smplx_dict)

        output = body_model(**smplx_dict)
        vertices = output.vertices.detach().cpu().numpy()
        joints = output.joints.detach().cpu().numpy()

        """
        body_meshes = []
        for seq_idx in range(vertices.shape[0]):
            m = trimesh.Trimesh(vertices=vertices[seq_idx], faces=body_model.faces, \
                                visual=trimesh.visual.TextureVisuals(uv=uv, image=body_texture[smpl_texture_path]), process=False)
            body_meshes.append(m)
        body_mesh = pyrender.Mesh.from_trimesh(body_meshes, smooth=False)
        # viewer.render_lock.acquire()
        if body_node is not None:
            scene.remove_node(body_node)
        body_node = pyrender.Node(mesh=body_mesh, name='body')
        scene.add_node(body_node)
        # viewer.render_lock.release()
        """

        for seq_idx in range(joints.shape[0]):
            flag_2d_joint = False
            flag_visible = False

            # only for one AR
            joint = joints[seq_idx]
            # 57: leye 56: reye
            # look_front. approx. may not be vertical to look_right
            look_at = joint[57] - joint[23] + joint[56] - joint[24]
            look_at = look_at.astype(np.float64)
            look_at = look_at / np.linalg.norm(look_at)
            # look_right
            leye_reye_dir = joint[23] - joint[24] 
            leye_reye_dir = leye_reye_dir.astype(np.float64)
            leye_reye_dir = leye_reye_dir / np.linalg.norm(leye_reye_dir)
            # look_up
            look_up_dir = np.cross(leye_reye_dir, look_at) 
            look_up_dir = look_up_dir.astype(np.float64)
            look_up_dir /= np.linalg.norm(look_up_dir)
            # only keep vertical componenet of look_at 
            look_at = np.cross(look_up_dir, leye_reye_dir)
            look_at = look_at.astype(np.float64)
            look_at /= np.linalg.norm(look_at)

            cam_pos = (joint[23] + joint[24]) / 2.
            # viewer.render_lock.acquire()
            if camera_node is not None:
                scene.remove_node(camera_node)
            up = np.array([0,1,0])
            front = np.array([0,0,-1])
            right = np.cross(up, front)
            look_at_up = np.cross(look_at, leye_reye_dir)
            look_at_up = look_at_up.astype(np.float64)
            look_at_up /= np.linalg.norm(look_at_up)
            r1 = np.stack([leye_reye_dir, look_at_up, look_at])
            r2 = np.stack([right, up, front])
            quat = pyquat.Quaternion(matrix=(r1.T @ r2))
            quat_pyrender = [quat[1], quat[2], quat[3], quat[0]]
            camera_node = pyrender.Node(camera=camera, name='camera', rotation=quat_pyrender, translation=cam_pos)
            scene.add_node(camera_node)
            # viewer.render_lock.release()

            # interactee joints projected to 2d camera plane, keep frames with >=6 joints visible
            interactee_joints_3d = joints[1-seq_idx][0:22]  # [22, 3] select 22 main body joints

            # discard all back-to-back frames
            look_at_2d = look_at[:2].astype(np.float64)
            look_at_2d /= np.linalg.norm(look_at_2d)
            dir_to_interactee = interactee_joints_3d[0][:2] - cam_pos[:2]
            dir_to_interactee = dir_to_interactee.astype(np.float64)
            dir_to_interactee /= np.linalg.norm(dir_to_interactee)
            if np.arccos(np.clip(np.dot(look_at_2d, dir_to_interactee), -1.0, 1.0)) < np.pi / 2:
                flag_visible = True
            else:
                continue

            # project 3d joints to depth camera 2d plane
            cam_intrinsics = np.array([[fx, 0., cx],
                                     [0., fx, cy],
                                     [0., 0., 1.]])
            camera_pose[:3, :3] = r1.T @ r2
            camera_pose[:3, 3] = cam_pos
            Rt = np.linalg.inv(camera_pose)
            interactee_joints_2d = cv2.projectPoints(interactee_joints_3d,
                                                     cv2.Rodrigues(Rt[:3, :3])[0], Rt[:3, 3],
                                                     cam_intrinsics,
                                                     np.array([[0.0, 0, 0, 0, 0, 0, 0, 0]]))[0].squeeze()  # [22, 2]
            valid_x = np.logical_and(interactee_joints_2d[:, 1] >= 0, interactee_joints_2d[:, 1] <= 1080)
            valid_y = np.logical_and(interactee_joints_2d[:, 0] >= 0, interactee_joints_2d[:, 0] <= 1920)
            valid_joint_num = np.sum(valid_x * valid_y)
            if valid_joint_num >= 6:
                flag_2d_joint = True
            else:
                continue

            if flag_dist and flag_2d_joint and flag_visible:
                
                m = make_new_mesh(smplx_vt, smplx_f, smplx_ft, vertices[1-seq_idx], body_texture[smpl_texture_path])

                # m = trimesh.Trimesh(vertices=vertices[1-seq_idx], faces=body_model.faces, \
                #                     visual=trimesh.visual.TextureVisuals(uv=uv, image=body_texture[smpl_texture_path]), process=False)

                # body_mesh = create_pbr_mesh(m, body_texture[smpl_texture_path], roughness=0.5)
                body_mesh = pyrender.Mesh.from_trimesh(m, smooth=True)
                if body_node is not None:
                    scene.remove_node(body_node)
                body_node = pyrender.Node(mesh=body_mesh, name='body')
                scene.add_node(body_node)

                # m = trimesh.Trimesh(vertices=pant[1-seq_idx]['verts'][frame_idx - 2], faces=pant[1-seq_idx]['faces'],\
                #                     visual=trimesh.visual.TextureVisuals(uv=pant_uv, image=clothing_texture[clothing_name][clothing_texture_name]), process=False)
                m = make_new_mesh(pant_vt, pant_f, pant_ft, pant[1-seq_idx]['verts'][frame_idx - 2], clothing_texture[clothing_name][clothing_texture_name])

                # pant_mesh = create_pbr_mesh(m, clothing_texture[clothing_name][clothing_texture_name], roughness=0.5)
                pant_mesh = pyrender.Mesh.from_trimesh(m, smooth=True)
                if pant_node is not None:
                    scene.remove_node(pant_node)
                pant_node = pyrender.Node(mesh=pant_mesh, name='pant')
                scene.add_node(pant_node)

                # m = trimesh.Trimesh(vertices=top[1-seq_idx]['verts'][frame_idx - 2], faces=top[1-seq_idx]['faces'],\
                #                     visual=trimesh.visual.TextureVisuals(uv=top_uv, image=clothing_texture[clothing_name][clothing_texture_name]), process=False)
                m = make_new_mesh(top_vt, top_f, top_ft, top[1-seq_idx]['verts'][frame_idx - 2], clothing_texture[clothing_name][clothing_texture_name])
                
                # top_mesh = create_pbr_mesh(m, clothing_texture[clothing_name][clothing_texture_name], roughness=0.5)
                top_mesh = pyrender.Mesh.from_trimesh(m, smooth=True)
                if top_node is not None:
                    scene.remove_node(top_node)
                top_node = pyrender.Node(mesh=top_mesh, name='top')
                scene.add_node(top_node)

                # check if body is visible in the image
                nm = {body_node: 77, top_node: 77, pant_node: 77, object_node: 20}
                seg, _ = renderer.render(scene, 8192, nm)
                if np.where(seg[:, :, 0] == 77)[0].shape[0] < 20000:
                    # ignore all imgs with less than 20k human pixels
                    continue

                rgb, depth = renderer.render(scene)

                global valid_num
                valid_num += 1
                # TODO: 1-seq_idx only correct for 2 human scenario
                base_path = 'tmp/egobody_rgb'
                # if not os.path.exists(os.path.join(base_path, scene_name, 'rgb')):
                #     os.makedirs(os.path.join(base_path, scene_name, 'rgb'))
                if not os.path.exists(os.path.join(base_path, scene_name, 'rgb')):
                    os.makedirs(os.path.join(base_path, scene_name, 'rgb'))
                if not os.path.exists(os.path.join(base_path, scene_name, 'smplx_params')):
                    os.makedirs(os.path.join(base_path, scene_name, 'smplx_params'))
                # np.save(os.path.join(base_path, scene_name, 'rgb', '%d.npy' % valid_num), rgb)
                cv2.imwrite(os.path.join(base_path, scene_name, 'rgb', '%d.jpg' % valid_num), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                custom_smplx_params = np.zeros(99)
                custom_smplx_params[:69] = smplx_params[1 - seq_idx, frame_idx, :69]
                custom_smplx_params[69:85] = Rt.reshape(-1)
                custom_smplx_params[85:95] = betas[1 - seq_idx]
                custom_smplx_params[95] = 0 if genders[1 - seq_idx] == 'male' else 1
                custom_smplx_params[96] = cx
                custom_smplx_params[97] = cy
                custom_smplx_params[98] = fx
                np.save(os.path.join(base_path, scene_name, 'smplx_params', '%d.npy' % valid_num), custom_smplx_params)



model_path = "data/smplx/models"
body_texture_path = np.load('HOOD/hood_data/bedlam/body_texture_path.npy')
clothing_texture_paths = np.load('HOOD/hood_data/bedlam/new_clothing_texture_path.npy')
HOOD_PYTHON = "/home/amey/miniconda3/envs/hood/bin/python"
smplx_vt = obj_vt('HOOD/hood_data/bedlam/smplx_uv.obj')
smplx_f = obj_fv('HOOD/hood_data/bedlam/smplx_uv.obj')
smplx_ft = obj_ft('HOOD/hood_data/bedlam/smplx_uv.obj')
device = torch.device('cuda')
valid_num = 0

bm_male20 = smplx.create(model_path=model_path,
                          model_type='smplx',
                          gender="male",
                          use_pca=False,
                          batch_size=20,
                          ).to(device='cuda')
bm_female20 = smplx.create(model_path=model_path,
                          model_type='smplx',
                          gender="female",
                          use_pca=False,
                          batch_size=20,
                          ).to(device='cuda')

bm_male2 = smplx.create(model_path=model_path,
                          model_type='smplx',
                          gender="male",
                          use_pca=False,
                          batch_size=2,
                          ).to(device)
bm_female2 = smplx.create(model_path=model_path,
                          model_type='smplx',
                          gender="female",
                          use_pca=False,
                          batch_size=2,
                          ).to(device)

body_texture = {}
for tex_path in body_texture_path:
    # im = Image.alpha_composite(Image.open(tex_path), eye_img)
    # im.save("/mnt/vlg-nfs/genli/datasets/bedlam/" + tex_path.split('/')[-1])
    body_texture[tex_path.split('/')[-1]] = Image.open(tex_path)

clothing_texture = {}
# for clothing_name in ['rp_aaron_posed_009', "rp_aaron_posed_013", "rp_ethan_posed_015", "rp_henry_posed_001", "rp_alexandra_posed_025", "rp_aneko_posed_011", "rp_claudia_posed_020"]:
for clothing_name in ['rp_aaron_posed_009', "rp_aaron_posed_013", "rp_alexandra_posed_025", "rp_aneko_posed_011"]: 
    clothing_texture[clothing_name] = {}
for tex_path in clothing_texture_paths:
    clothing_name = tex_path.split('/')[-4]
    clothing_texture[clothing_name][tex_path.split('/')[-2]] = Image.open(tex_path)

# if __name__ == '__main__':
def genRGB(max_num, scene_name):
    # make sure you installed hood conda env first!
    # HOOD_PYTHON = "/home/amey/miniconda3/envs/hood/bin/python"
    if "genli" in HOOD_PYTHON:
        print()
        print("     WARNING: Please replace HOOD_PYTHON with your hood python path!")
        print("     You can find it by: ")
        print("         conda activate hood")
        print("         which python")
        print("         conda deactivate")
        print()
        pdb.set_trace()
    # MAX_NUM = 20000
    MAX_NUM = max_num
    # with open('/mnt/vlg-nfs/kaizhao/datasets/scene_mesh_4render/list.txt', 'r') as f:
    #     scene_names = f.readlines()

    # for scene_name in scene_names:
    if True:
        # scene_name = scene_name.strip()
        # scene_name = "seminar_d78"
        # valid_num = 0
        scene_mesh = trimesh.load(os.path.join('exp_data', scene_name, 'mesh_floor_zup.ply'))
        # scene_mesh = trimesh.load(os.path.join('exp_data', scene_name, 'cab_e.obj'))

        # smplx_uv = trimesh.load('/mnt/vlg-nfs/genli/datasets/bedlam/smplx_uv.obj', process=False)
        # smplx_uv.merge_vertices(merge_tex=True)
        # uv = smplx_uv.visual.uv
        # smplx_vt = obj_vt('HOOD/hood_data/bedlam/smplx_uv.obj')
        # smplx_f = obj_fv('HOOD/hood_data/bedlam/smplx_uv.obj')
        # smplx_ft = obj_ft('HOOD/hood_data/bedlam/smplx_uv.obj')

        # body_texture_path = np.load('HOOD/hood_data/bedlam/body_texture_path.npy')
        # clothing_texture_paths = np.load('HOOD/hood_data/bedlam/new_clothing_texture_path.npy')
        # eye_path = '/mnt/vlg-nfs/genli/bedlam/bedlam_body_textures_meshcapade/eye/SMPLX_eye.png' 
        # eye_img = Image.open(eye_path)
        # body_texture = {}
        # for tex_path in body_texture_path:
        #     # im = Image.alpha_composite(Image.open(tex_path), eye_img)
        #     # im.save("/mnt/vlg-nfs/genli/datasets/bedlam/" + tex_path.split('/')[-1])
        #     body_texture[tex_path.split('/')[-1]] = Image.open(tex_path)
        
        # clothing_texture = {}
        # # for clothing_name in ['rp_aaron_posed_009', "rp_aaron_posed_013", "rp_ethan_posed_015", "rp_henry_posed_001", "rp_alexandra_posed_025", "rp_aneko_posed_011", "rp_claudia_posed_020"]:
        # for clothing_name in ['rp_aaron_posed_009', "rp_aaron_posed_013", "rp_alexandra_posed_025", "rp_aneko_posed_011"]: 
        #     clothing_texture[clothing_name] = {}
        # for tex_path in clothing_texture_paths:
        #     clothing_name = tex_path.split('/')[-4]
        #     clothing_texture[clothing_name][tex_path.split('/')[-2]] = Image.open(tex_path)
        
        while True:
            gen_data_egobody(
                scene_mesh = scene_mesh,
                vis_navmesh=False,
                vis_marker=False, vis_pelvis=False, vis_object=True, add_floor=False, scene_name=scene_name,
                )
            if valid_num >= MAX_NUM:
                break
