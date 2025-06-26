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



## Simplicity, User-Friendliness
image = load_image("./image/real/0.png")

tgt_landmark_path = './dataset/base_image_dataset/0/0_landmark.png'
tgt_face_prompt = 'a very surprised expression, front view'

face_token = pipe.get_edited_token(image,tgt_face_prompt,tgt_landmark_path)

pure_prompt = ['a woman walking on the street','']
negtive_prompt_group="(nude:1.0), (naked:1.0), (bare breasts:1.0), (nipples:1.0),((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"
negative_prompt = negtive_prompt_group

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
    activate_query=True,
    face_tokens = face_token,
    infer_end = 0,
    infer_scale= 0,
).images[0]

     
## Optimal Effectiveness
image = load_image("./image/real/0.png")
src_face_prompt = 'a woman with short black hair, smiling with teeth showing, looking forward, face slightly turned to the side.'
tgt_landmark_path = './dataset/base_image_dataset/0/0_landmark.png'
tgt_face_prompt = 'a very surprised expression, front view'

face_token = pipe.get_edited_token(image, tgt_face_prompt, tgt_landmark_path, src_face_prompt)

pure_prompt = ['a woman walking on the street']
negtive_prompt_group="(nude:1.0), (naked:1.0), (bare breasts:1.0), (nipples:1.0),((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"
negative_prompt = negtive_prompt_group

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
    activate_query=True,
    face_tokens = face_token,
    infer_end = 0,
    infer_scale= 0,
).images[0]


## layout control
image = load_image("./image/real/0.png")
src_face_prompt = 'a woman with short black hair, smiling with teeth showing, looking forward, face slightly turned to the side.'
tgt_landmark_path = './dataset/base_image_dataset/0/0_landmark.png'
tgt_face_prompt = 'a very surprised expression, front view'

face_token = pipe.get_edited_token(image, tgt_face_prompt, tgt_landmark_path, src_face_prompt)

pure_prompt = ['a woman walking on the street']
negtive_prompt_group="(nude:1.0), (naked:1.0), (bare breasts:1.0), (nipples:1.0),((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"
negative_prompt = negtive_prompt_group

# Position the character in the top left corner.
bbox = [[0.0,0.0,0.3,0.3]]

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
    activate_query=True,
    face_tokens = face_token,
    bbox = bbox,
    infer_end = 12,
    infer_scale= 2,
).images[0]


## context decoupling
names = [str(i)+'.png' for i in range(20)]
for name in names:
    image = load_image('./image/real/'+name+'.png')
    tgt_landmark_path = './dataset/base_image_dataset/0/0_landmark.png'
    tgt_face_prompt = 'a very surprised expression, front view'

    face_token = pipe.get_edited_token(image, tgt_face_prompt, tgt_landmark_path)

    pure_prompt = ['a person walking on the street']
    negtive_prompt_group="(nude:1.0), (naked:1.0), (bare breasts:1.0), (nipples:1.0),((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"
    negative_prompt = negtive_prompt_group



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
        activate_query=True,
        face_tokens = face_token,
        bbox = None,
        infer_end = 0,
        infer_scale= 0,
    ).images[0]
    
    
## identity mix
image1 = load_image("./image/real/0.png")
image2 = load_image("./image/real/1.png")


tgt_landmark_path = './dataset/base_image_dataset/0/0_landmark.png'
tgt_face_prompt = 'a happy expression, front view'

face_token1 = pipe.get_edited_token(image1,tgt_face_prompt,tgt_landmark_path)
face_token2 = pipe.get_edited_token(image2,tgt_face_prompt,tgt_landmark_path)
face_token = torch.cat([face_token1,face_token2],dim=1)

pure_prompt = ['a person in garden']
negtive_prompt_group="(nude:1.0), (naked:1.0), (bare breasts:1.0), (nipples:1.0),((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"
negative_prompt = negtive_prompt_group

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
    face_tokens = face_token,
    infer_end = 0,
    infer_scale= 0,
    mix_scale = 0.5,
).images[0]