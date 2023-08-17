import os
import io
import sys
import random
import cv2
import torch
import math
import customtkinter
import numpy as np
from pathlib import Path
from PIL import Image

customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

app = customtkinter.CTk()
app.geometry("400x200")
app.title("AI Mesh Creator")

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
    os.system("cls")
    iteration_number = random.randint(1, 9999999999999999999)
    os.makedirs(os.path.join(os.path.dirname(__file__), (prompt.replace(" ", "_") + str(iteration_number))))
    prompt_path = os.path.join(os.path.dirname(__file__), (prompt.replace(" ", "_") + str(iteration_number)))
    os.makedirs(os.path.join(prompt_path, "Depth_Map"))
    os.makedirs(os.path.join(prompt_path, "Generated_Image"))
    os.makedirs(os.path.join(prompt_path, "Masked_Image"))
    os.makedirs(os.path.join(prompt_path, "Mesh"))
    # os.makedirs(os.path.join(prompt_path, "Normal_Map"))
    
    # GENERATE AI IMAGE

    from diffusers import StableDiffusionPipeline
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    os.system("cls")
    image = pipe(prompt).images[0]
    os.chdir(os.path.join(prompt_path, "Generated_Image"))
    rgb_image_name = (prompt.replace(" ", "_") + "_" + str(iteration_number) + ".png")
    rgb_image_path = os.path.join(prompt_path, "Generated_Image", rgb_image_name)
    image.save(rgb_image_path)
    
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
    
    # TRIM MAIN SUBJECTS
    
    # Load paths and parameters
    # depth_image_path = "path_to_depth_image.jpg"
    # prompt_path = "output_directory"
    # prompt = "example_prompt"
    # iteration_number = 1

    # Load depth map
    # depth_map = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
    # threshold_value = 170

    # Threshold the depth map
    # _, binary_map = cv2.threshold(depth_map, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours and create mask
    # contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # mask = np.zeros_like(depth_map, dtype=np.uint8)
    # cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Create 4-channel masked depth map
    # masked_depth_map = cv2.merge([depth_map, mask])

    # Save masked depth map with transparency
    # masked_depth_image_path = os.path.join(prompt_path, "Depth_Map", (prompt.replace(" ", "_") + "_" + str(iteration_number) + "_masked.png"))
    # cv2.imwrite(masked_depth_image_path, masked_depth_map)
    
    # SET PATHS FOR MODEL ETC
    
    depthPath = depth_image_path
    texturePath = rgb_image_path
    # depthPath = masked_depth_image_path
    # texturePath = rgb_image_path
    objPath = os.path.join(prompt_path, "Mesh", (prompt.replace(" ", "_") + "_" + str(iteration_number) + ".obj"))
    mtlPath = os.path.join(prompt_path, "Mesh", (prompt.replace(" ", "_") + "_" + str(iteration_number) + ".mtl"))
    matName = (prompt.replace(" ", "_") + "_" + str(iteration_number))

    # MODEL FROM DEPTH AND RGB	
    
    generate_model(depthPath, texturePath, objPath, mtlPath, matName)
    torch.cuda.empty_cache()
def button_callback():
    prompt = text_1.get("1.0", "end-1c")
    generate(prompt)

frame_1 = customtkinter.CTkFrame(master=app)
frame_1.pack(pady=20, padx=20, fill="both", expand=True)

text_1 = customtkinter.CTkTextbox(master=frame_1, width=200, height=70)
text_1.pack(pady=10, padx=10)
text_1.insert("0.0", "Enter Prompt Here:")

button_1 = customtkinter.CTkButton(master=frame_1, text="Generate", command=button_callback)
button_1.pack(pady=10, padx=10)

app.mainloop()
