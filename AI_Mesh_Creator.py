import bpy
import math
import os
import sys
import subprocess
import atexit
import string
import random
import os.path

bl_info = {
    "name": "AI Mesh Creator",
    "author": "Joshua Roberts",
    "version": (1, 0),
    "blender": (3, 5, 0),
    "location": "View3D > AI_Mesh",
    "description": "Generate 3D models with prompts",
    "category": "AI Mesh",
}

class GenerateModelAddonPanel(bpy.types.Panel):
    bl_label = "AI Mesh Creator"
    bl_idname = "PT_GENERATE_MODEL"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Generate'

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="Prompt:")
        box.prop(context.scene, "generate_prompt", text="")
        box.operator("object.generate_model", text="Generate")

class OBJECT_OT_GenerateModel(bpy.types.Operator):
    bl_idname = "object.generate_model"
    bl_label = "Generate Model"
    
    def execute(self, context):
        prompt = context.scene.generate_prompt
        generate(prompt)
        return {'FINISHED'}

def generate_model(depthPath, texturePath, objPath, mtlPath, matName):
	def create_mtl(mtlPath, matName, texturePath):
		if max(mtlPath.find('\\'), mtlPath.find('/')) > -1:
			os.makedirs(os.path.dirname(mtlPath), exist_ok=True)
	with open(mtlPath, "w") as f:
		f.write("newmtl " + matName + "\n"      )
		f.write("Ns 10.0000\n"                  )
		f.write("d 1.0000\n"                    )
		f.write("Tr 0.0000\n"                   )
		f.write("illum 2\n"                     )
		f.write("Ka 1.000 1.000 1.000\n"        )
		f.write("Kd 1.000 1.000 1.000\n"        )
		f.write("Ks 0.000 0.000 0.000\n"        )
		f.write("map_Ka " + texturePath + "\n"  )
		f.write("map_Kd " + texturePath + "\n"  )

	def vete(v, vt):
		return str(v)+"/"+str(vt)

	def create_obj(depthPath, objPath, mtlPath, matName):
    
		img = cv2.imread(depthPath,cv2.IMREAD_GRAYSCALE).astype(np.float32) / 1000.0
		img = 1.0 - img

		w = img.shape[1]
		h = img.shape[0]

		FOV = math.pi/4
		D = (img.shape[0]/2)/math.tan(FOV/2)

		if max(objPath.find('\\'), objPath.find('/')) > -1:
			os.makedirs(os.path.dirname(mtlPath), exist_ok=True)
    
		with open(objPath,"w") as f:    
			f.write("mtllib " + mtlPath + "\n")
			f.write("usemtl " + matName + "\n")

			ids = np.zeros((img.shape[1], img.shape[0]), int)
			vid = 1

			for u in range(0, w):
				for v in range(h-1, -1, -1):

					d = img[v, u]

					ids[u,v] = vid
					if d.all() == 0.0:
						ids[u,v] = 0
					vid += 1

					x = u - w/2
					y = v - h/2
					z = -D

					norm = 1 / math.sqrt(x*x + y*y + z*z)

					t = d/(z*norm)

					x = -t*x*norm
					y = t*y*norm
					z = -t*z*norm        

					f.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")

			for u in range(0, img.shape[1]):
				for v in range(0, img.shape[0]):
					f.write("vt " + str(u/img.shape[1]) + " " + str(v/img.shape[0]) + "\n")

			for u in range(0, img.shape[1]-1):
				for v in range(0, img.shape[0]-1):

					v1 = ids[u,v]; v2 = ids[u+1,v]; v3 = ids[u,v+1]; v4 = ids[u+1,v+1];

					if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
						continue

					f.write("f " + vete(v1,v1) + " " + vete(v2,v2) + " " + vete(v3,v3) + "\n")
					f.write("f " + vete(v3,v3) + " " + vete(v2,v2) + " " + vete(v4,v4) + "\n")

	print("STARTED")
	create_mtl(mtlPath, matName, texturePath)
	create_obj(depthPath, objPath, mtlPath, matName)
	print("FINISHED")



def generate(prompt):
    from pathlib import Path
    from PIL import Image
    import numpy as np
    import requests
    import torch
    import cv2
    import threading
    import argparse
    iteration_number = random.randint(1, 9999999999999999999)
    os.makedirs(os.path.join(os.path.dirname(__file__), (prompt.replace(" ", "_") + str(iteration_number))))
    prompt_path = os.path.join(os.path.dirname(__file__), (prompt.replace(" ", "_") + str(iteration_number)))
    os.makedirs(os.path.join(prompt_path, "Depth_Map"))
    os.makedirs(os.path.join(prompt_path, "Generated_Image"))
    os.makedirs(os.path.join(prompt_path, "Mesh"))
    os.makedirs(os.path.join(prompt_path, "Normal_Map"))



    # GENERATE AI IMAGE

    torch.cuda.empty_cache()
    from diffusers import StableDiffusionPipeline
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(prompt).images[0]
    os.chdir(os.path.join(prompt_path, "Generated_Image"))
    rgb_image_name = (prompt.replace(" ", "_") + "_" + str(iteration_number) + ".png")
    rgb_image_path = os.path.join(prompt_path, "Generated_Image", rgb_image_name)
    image.save(rgb_image_path)
    


    # NORMAL MAP

    torch.cuda.empty_cache()
    from PIL import ImageFilter
    input_image_path = rgb_image_path
    input_image = Image.open(input_image_path)
    input_image = input_image.convert('RGB')
    gray_image = input_image.convert('L')
    edges = gray_image.filter(ImageFilter.FIND_EDGES)
    edges_array = (np.array(edges) / 255.0) * 2 - 1
    edges_array = edges_array[..., np.newaxis]
    normal_map = np.zeros((edges_array.shape[0], edges_array.shape[1], 3))
    normal_map[..., 2] = 1.0
    normal_map[..., 0] = edges_array[..., 0]
    normal_map_image = Image.fromarray((normal_map * 255).astype(np.uint8))
    normal_map_image_path = os.path.join(prompt_path, "Normal_Map", (prompt.replace(" ", "_") + "_" + str(iteration_number) + "_normal.png"))
    normal_map_image.save(normal_map_image_path)



    # MONOCULAR DEPTH EST

    torch.cuda.empty_cache()
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    image = Image.open(rgb_image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    os.chdir(os.path.join(prompt_path, "Depth_Map"))
    depth_image_name = (prompt + "_" + str(iteration_number) + ".png")
    depth_image_path = os.path.join(prompt_path, "Depth_Map", depth_image_name)
    depth.save(depth_image_path)
    depthPath = depth_image_path
    texturePath = rgb_image_path
    objPath = os.path.join(prompt_path, "Mesh", (prompt.replace(" ", "_") + "_" + str(iteration_number) + ".obj"))
    mtlPath = os.path.join(prompt_path, "Mesh", (prompt.replace(" ", "_") + "_" + str(iteration_number) + ".mtl"))
    matName = (prompt.replace(" ", "_") + "_" + str(iteration_number))



    # MODEL FROM DEPTH AND RGB	
    
    generate_model(depthPath, texturePath, objPath, mtlPath, matName)
    torch.cuda.empty_cache()



    # ADD NORMAL MAP TO MTL

    mtl_path = os.path.join(prompt_path, "Mesh", (prompt.replace(" ", "_") + "_" + str(iteration_number) + ".mtl"))
    material_name = (prompt.replace(" ", "_") + "_" + str(iteration_number))
    normal_map_path = normal_map_image_path
    with open(mtl_path, 'r') as f:
        mtl_content = f.read()
    material_start = mtl_content.find('newmtl ' + material_name)
    if material_start == -1:
        print(f"Material '{material_name}' not found in the .mtl file.")
        return
    normal_map_line = f"map_Bump {normal_map_path}\n"
    mtl_content = mtl_content[:material_start] + normal_map_line + mtl_content[material_start:]
    with open(mtl_path, 'w') as f:
        f.write(mtl_content)
    torch.cuda.empty_cache()



    # IMPORT GENERATED MODEL TO SCENE

    bpy.ops.import_scene.obj(filepath=objPath)
    imported_object = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = imported_object
    imported_object.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    imported_object.location = (0, 0, 0)
    smooth_factor = 2.5
    repeat_count = 20
    smooth_modifier = imported_object.modifiers.new(name="Smooth", type='SMOOTH')
    smooth_modifier.factor = smooth_factor
    smooth_modifier.iterations = repeat_count
    


    # ADD NORMAL MAP TO OBJECT MATERIAL

    selected_object = bpy.context.active_object
    if len(selected_object.data.materials) == 0:
        material = bpy.data.materials.new(name="Material")
        selected_object.data.materials.append(material)
    else:
        material = selected_object.data.materials[0]
    texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')
    texture_node.location = (0, 300)
    texture_node.image = bpy.data.images.load(normal_map_path)
    normal_map_node = material.node_tree.nodes.new(type='ShaderNodeNormalMap')
    normal_map_node.location = (200, 300)
    diffuse_shader_node = material.node_tree.nodes["Principled BSDF"]
    diffuse_shader_node.location = (400, 300)
    material.node_tree.links.new(texture_node.outputs['Color'], normal_map_node.inputs['Color'])
    material.node_tree.links.new(normal_map_node.outputs['Normal'], diffuse_shader_node.inputs['Normal'])
    material.node_tree.nodes.active = normal_map_node
    material.use_nodes = True


# PIP REQUIREMENTS

class PipRequirementsAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    def draw(self, context):
        layout = self.layout
        layout.label(text="Click 'Install Pip Requirements' to install:")
        layout.operator("addon.install_pip_requirements", text="Install Pip Requirements")

class InstallPipRequirementsOperator(bpy.types.Operator):
    bl_idname = "addon.install_pip_requirements"
    bl_label = "Install Pip Requirements"
    
    def execute(self, context):
        # Get Blender's Python executable and the path to the site-packages folder
        blender_python = sys.executable
        site_packages_path = os.path.join(os.path.dirname(blender_python), "lib", "site-packages")

        requirements = [
            "opencv-python",
            "transformers",
            "diffusers",
            "open3d",
            "pillow",
            "torch",
            "torchvision",
            "torchaudio",
        ]
        
        index_url = "https://download.pytorch.org/whl/cu118"
        
        def install_packages():
            total_requirements = len(requirements)
            
            for i, req in enumerate(requirements, 1):
                subprocess.run([blender_python, "-m", "pip", "install", req, "--index-url", index_url])
                self.report({'INFO'}, f"Installed {req} ({i}/{total_requirements})")
            
            self.report({'INFO'}, "Pip requirements installation completed.")
        
        # Run installation in a separate thread
        thread = threading.Thread(target=install_packages)
        thread.start()
        
        return {'RUNNING_MODAL'}


# Register

def register():
    bpy.utils.register_class(PipRequirementsAddonPreferences)
    bpy.utils.register_class(InstallPipRequirementsOperator)
    bpy.utils.register_class(GenerateModelAddonPanel)
    bpy.utils.register_class(OBJECT_OT_GenerateModel)
    bpy.types.Scene.generate_prompt = bpy.props.StringProperty(
        name="Prompt",
        description="Enter the prompt text",
        default="Your Prompt Here",
    )

def unregister():
    bpy.utils.unregister_class(PipRequirementsAddonPreferences)
    bpy.utils.unregister_class(InstallPipRequirementsOperator)
    bpy.utils.unregister_class(GenerateModelAddonPanel)
    bpy.utils.unregister_class(OBJECT_OT_GenerateModel)
    del bpy.types.Scene.generate_prompt

if __name__ == "__main__":
    register()
