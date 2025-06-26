import torch
import os
from diffusers.utils import load_image
from diffusers import DDIMScheduler
from pipeline import DynamicIDStableDiffusionPipeline



device = "cuda"
base_model_path = "./models/Realistic_Vision_V6.0_B1_noVAE"

pipe = DynamicIDStableDiffusionPipeline.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, 
).to(device)


pipe.load_DynamicID(
    SAA_path = './models/SAA.bin',
    IMR_path = './models/IMR.bin',
)     
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


## two identities
image1 = load_image("./image/real/0.png")
tgt_landmark_path = './dataset/base_image_dataset/0/0_landmark.png'
tgt_face_prompt = 'a happy expression, side view.'   
face_token1 = pipe.get_edited_token(image1,tgt_face_prompt,tgt_landmark_path)

image2 = load_image("./image/real/1.png")
tgt_landmark_path = './dataset/base_image_dataset/0/0_landmark.png' 
tgt_face_prompt  = 'a netural expression, front view.'  
face_token2 = pipe.get_edited_token(image2,tgt_face_prompt,tgt_landmark_path)

face_tokens = torch.cat([face_token1,face_token2],dim=1)
bbox = [[0.1,0.1,0.5,0.4],[0.1,0.6,0.5,0.9]]
pure_prompt = ['a portrait of two people, in bus']
negative_prompt="(nude:1.0), (naked:1.0), (bare breasts:1.0), (nipples:1.0),((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"


        
generator = torch.Generator(device=device).manual_seed(2025)
images = pipe(
    prompt=pure_prompt,
    width=768,    
    height=768,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=50,
    generator=generator,
    id_scale = 1,
    aware_query=True,
    bbox = bbox,
    face_tokens = face_tokens,
    infer_end = 12,
    infer_scale= 2,
).images[0]




## more identities
generator = torch.Generator(device=device).manual_seed(2025)

image1 = load_image("./image/real/0.png")
tgt_landmark_path = './dataset/base_image_dataset/0/0_landmark.png'
tgt_face_prompt = 'a happy expression, side view.'   
face_token1 = pipe.get_edited_token(image1,tgt_face_prompt,tgt_landmark_path)

image2 = load_image("./image/real/1.png")
tgt_landmark_path = './dataset/base_image_dataset/0/0_landmark.png' 
tgt_face_prompt  = 'a netural expression, front view.'  
face_token2 = pipe.get_edited_token(image2,tgt_face_prompt,tgt_landmark_path)

image3 = load_image("./image/real/2.png")
tgt_landmark_path = './dataset/base_image_dataset/0/0_landmark.png'
tgt_face_prompt = 'a happy expression, side view.'   
face_token3 = pipe.get_edited_token(image3,tgt_face_prompt,tgt_landmark_path)

face_tokens = torch.cat([face_token1,face_token2,face_token3],dim=1)
bbox = [[0,0,1,0.3],[0,0.4,1,0.6],[0,0.7,1,1]]
pure_prompt = ['three men wearing shirts, on the street, cinematic photo, portrait,film, professional, 4k, highly detailed']
negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality, blurry, nsfw, nudity"+"(nude:1.0), (naked:1.0), (bare breasts:1.0), (nipples:1.0),((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"


generator = torch.Generator(device=device).manual_seed(2025)
images = pipe(
    prompt=pure_prompt,
    width=1024,    
    height=1024,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=50,
    generator=generator,
    id_scale = 1,
    aware_query=True,
    bbox = bbox,
    face_tokens = face_tokens,
    infer_end = 12,
    infer_scale= 2,
    identity_remove_step = 0,
).images[0]

    
    