import torch
import os
from pipeline import DynamicIDStableDiffusionPipeline
from diffusers.utils import load_image




device = "cuda"


base_model_path = "./models/Realistic_Vision_V6.0_B1_noVAE"
SAA_path = "./models/SAA.bin" 


pipe = DynamicIDStableDiffusionPipeline.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, 
).to(device)


pipe.load_DynamicID(SAA_path)     


root_path = "./dataset/base_image_dataset"
num = len(os.listdir(root_path))
for i in range(num):
    person_path = os.path.join(root_path, str(i))
    image_path = sorted(os.listdir(person_path))
    
    save_path = person_path.replace('base_image_dataset','cache')
    os.makedirs(save_path, exist_ok=True)
    
    for path in image_path:
        if path.endswith('.txt') or path.endswith('landmark.png'):
            continue
        select_image = load_image(os.path.join(person_path, path))
        face_embed = pipe.cal_face_embed(select_image)
        if face_embed is None:
            print(f"Error in {str(i)}_{path}")           
        else:
            torch.save(face_embed,os.path.join(save_path, path.split('.')[0]+'.pt'))
    
