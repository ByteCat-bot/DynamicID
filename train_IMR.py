import os
import torch
import torch.nn.functional as F
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm 
from attention import Consistent_IPAttProcessor, Consistent_AttProcessor
from model import IMR,KeyPointEncoder
import itertools
from torchvision import transforms
from PIL import Image
import random
import copy
import argparse
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, root_path = './dataset/',  standard_size = 32):
        super().__init__()
        self.image_root_path = os.path.join(root_path, 'base_image_dataset')
        self.token_root_path = os.path.join(root_path, 'cache','base_cache')
        self.standard_size = standard_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        
        ## for KDEF dataset
        exp = ['afraid','angry','disgusted','happy','neutral','sad','surprised']
        pose = [
            'fully turned to the left',
            'fully turned to the right',
            'slightly turned to the left',
            'slightly turned to the right',
            'front view'
            ]
        
        self.face_prompts=[]
        for e in exp:
            for p in pose:
                self.face_prompts.append('the expression is '+ e +', the face is '+p)
        

    def get_valid_data_id(self, idx, ids):
        id = random.sample(ids, 1)[0]
        landmark_path = os.path.join(self.image_root_path, str(idx), id+'_landmark.png')
        if os.path.exists(landmark_path):
            return id
        else:
            return self.get_valid_data_id(idx,ids)
    
    
    def KDEF_data(self):
        img_root_path = './dataset/KDEF'
        token_root_path = './dataset/cache/KDEF_cache'
        person_names = sorted(os.listdir(token_root_path))
        person = random.sample(person_names, 1)[0]
        img_root_path = os.path.join(img_root_path, person)
        token_root_path = os.path.join(token_root_path, person)
        img_names = sorted(os.listdir(token_root_path))
        img_names = [img_name.split('.')[0] for img_name in img_names]
        tokens = []
        images = []
        landmarks = []
        face_prompts = self.face_prompts
        for img_name in img_names:
            token = torch.load(os.path.join(token_root_path, img_name+'.pt'),map_location='cpu')
            raw_image = Image.open(os.path.join(img_root_path, img_name+'.jpg'))
            image = self.transform(raw_image.convert("RGB"))
            landmark_image = Image.open(os.path.join(img_root_path, img_name+'_landmark.png')).convert('L')
            landmark_image = self.transform(landmark_image)
            tokens.append(token)
            images.append(image)
            landmarks.append(landmark_image)
        src_id = random.sample([i for i in range(len(tokens))], 1)[0]
        src_token = copy.deepcopy(tokens[src_id])
        src_face_prompt = copy.deepcopy(face_prompts[src_id])
        src_landmark = copy.deepcopy(landmarks[src_id]).unsqueeze(0)
        tgt_tokens = tokens[:src_id] + tokens[src_id+1:]
        tgt_images = images[:src_id] + images[src_id+1:]
        tgt_face_prompts = face_prompts[:src_id] + face_prompts[src_id+1:]
        tgt_landmarks = landmarks[:src_id] + landmarks[src_id+1:]
        tgt_tokens = torch.cat(tgt_tokens, dim=0)
        tgt_landmarks = torch.stack(tgt_landmarks, dim=0)
        tgt_images = torch.stack(tgt_images, dim=0)
        return {
            'source_token': src_token,
            'source_face_prompt':src_face_prompt,
            'source_landmark':src_landmark,
            'target_tokens': tgt_tokens,
            'target_images': tgt_images,
            'target_face_prompts': tgt_face_prompts,
            'target_landmarks':tgt_landmarks
        }
        
    def __getitem__(self, idx):
        rand_num = random.random()
        if rand_num<0.05:
            return self.KDEF_data()
        token_ids = os.listdir(os.path.join(self.token_root_path, str(idx)))
        ids = [id.split('.')[0] for id in token_ids]

        source_id = self.get_valid_data_id(idx,ids)

        source_token = torch.load(os.path.join(self.token_root_path, str(idx), source_id + '.pt'),map_location='cpu')
        source_face_prompt_path = os.path.join(self.image_root_path, str(idx), source_id+'.txt')
        with open(source_face_prompt_path, 'r') as f:
            source_face_prompt = f.read()
        source_landmark = Image.open(os.path.join(self.image_root_path, str(idx), source_id+'_landmark.png')).convert('L')
        source_landmark = self.transform(source_landmark).unsqueeze(0)
        target_images = []
        target_tokens = []
        target_face_prompts = []
        target_landmarks = []
        for i in range(self.standard_size):
            id = self.get_valid_data_id(idx,ids)
            target_token = torch.load(os.path.join(self.token_root_path, str(idx), id + '.pt'),map_location='cpu')
            raw_image = Image.open(os.path.join(self.image_root_path, str(idx), id + '.png'))
            target_face_prompt_path = os.path.join(self.image_root_path, str(idx), id+'.txt')
            landmark_image = Image.open(os.path.join(self.image_root_path, str(idx), id+'_landmark.png')).convert('L')
            landmark_image = self.transform(landmark_image)
            target_landmarks.append(landmark_image)
            with open(target_face_prompt_path, 'r') as f:
                target_face_prompt = f.read()
                target_face_prompts.append(target_face_prompt)
            target_tokens.append(target_token)
            target_images.append(self.transform(raw_image.convert("RGB")))
        target_tokens = torch.cat(target_tokens, dim=0)
        target_landmarks = torch.stack(target_landmarks, dim=0)
        target_images = torch.stack(target_images, dim=0)
        return {
            'source_token': source_token,
            'source_face_prompt':source_face_prompt,
            'source_landmark':source_landmark,
            'target_tokens': target_tokens,
            'target_images': target_images,
            'target_face_prompts': target_face_prompts,
            'target_landmarks':target_landmarks
        }
        
        
        
        
    
    def __len__(self):
        return len(os.listdir(self.image_root_path))
    

def collate_fn(datas):
    return datas[0]





class DynamicID(torch.nn.Module):
    def __init__(self, unet, device, weight_dtype):
        super().__init__()
        lora_rank = 128
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = Consistent_AttProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank,
                ).to(device, dtype=weight_dtype)
            else:
                attn_procs[name] = Consistent_IPAttProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0, rank=lora_rank, num_tokens=4,
                ).to(device, dtype=weight_dtype)
        
        unet.set_attn_processor(attn_procs)
        
        
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values()) 
        state_dict = torch.load("./SAA.bin",map_location="cpu")['adapter_modules']
        adapter_modules.load_state_dict(state_dict, strict=True)
        adapter_modules.requires_grad_(False)
        unet.requires_grad_(False)
        self.unet = unet


    def forward(self, noisy_latents, timesteps, prompt_embeds, faceid_tokens): 
        prompt_id_embeds = torch.cat([prompt_embeds, faceid_tokens], dim=1)
        noise_pred = self.unet(noisy_latents, timesteps, prompt_id_embeds).sample 
        return noise_pred


       
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--IMR_depth",
        type=int,
        default=1,
        help=(
            "The depth of an IMR network."
        ),
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=16,
        help="Number of tokens to query from the CLIP image encoding.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    




def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
    device = accelerator.device
    
    weight_dtype = args.weight_dtype
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_train_epochs = args.num_train_epochs
    batch_size=args.batch_size
    depth = args.depth
    save_steps = args.save_steps
    accumulation_steps = args.accumulation_steps
    dataloader_num_workers = args.dataloader_num_workers
    pretrained_model_path = args.pretrained_model_path
    common_prompt = 'portrait, cinematic photo, film, professional, 4k, highly detailed'

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet").to(device)
 
    text_inputs = tokenizer(
                common_prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
    common_prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0].detach().requires_grad_(False).to(weight_dtype)
    

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)




    unet.to(device, dtype=weight_dtype) 
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)    
    DynamicID_model = DynamicID(unet,device, weight_dtype=weight_dtype)



    imr = IMR(erase_layer_num=depth, drive_layer_num=depth).to(device, dtype=weight_dtype).requires_grad_(True)
    keypoint_encoder = KeyPointEncoder().to(device, dtype=weight_dtype).requires_grad_(True)
    
    params_to_opt = itertools.chain(keypoint_encoder.parameters(),  imr.parameters())
    
    optimizer = torch.optim.AdamW(params_to_opt, lr=learning_rate, weight_decay=weight_decay)

    # dataloader
    train_dataset = MyDataset()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=dataloader_num_workers,
    )
    
    keypoint_encoder, imr, optimizer, train_dataloader = accelerator.prepare(keypoint_encoder, imr, optimizer, train_dataloader)



    for epoch in range(num_train_epochs):
        LDC_loss_sum = 0
        DFM_loss_sum = 0
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                target_images = batch["target_images"].reshape(-1,3,512,512)
                latents = vae.encode(target_images.to(device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            

            source_token = batch["source_token"].to(device, dtype=weight_dtype)
            source_face_prompt = batch["source_face_prompt"]
            target_tokens = batch['target_tokens'].to(device, dtype=weight_dtype)
            target_face_prompts = batch['target_face_prompts']
            face_prompts = [source_face_prompt]+target_face_prompts
            text_inputs = tokenizer(
                face_prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            face_prompts = text_encoder(text_inputs.input_ids.to(device))[1].detach().requires_grad_(False).to(weight_dtype)
            src_face_prompt = face_prompts[:1]
            tgt_face_prompt = face_prompts[1:]
            
            src_landmark = batch['source_landmark']
            tgt_landmark = batch['target_landmarks']
            landmark_image = torch.cat([src_landmark,tgt_landmark],dim=0).to(device, dtype=weight_dtype)
            landmark_feature = keypoint_encoder(landmark_image)
            landmark_feature = landmark_feature.reshape(bsz+1,-1)
            
            source_landmark_feature = landmark_feature[:1]
            target_landmark_feature = landmark_feature[1:]
            prompt_embeds = common_prompt_embeds.repeat(bsz, 1, 1)
            
            
            pred_tokens = imr(source_token, source_landmark_feature, target_landmark_feature, src_face_prompt, tgt_face_prompt)

            tar_noise_pred = DynamicID_model(noisy_latents, timesteps, prompt_embeds, target_tokens)
            noise_pred = DynamicID_model(noisy_latents, timesteps, prompt_embeds, pred_tokens)
            

            DFM_loss = F.mse_loss(pred_tokens.float(), target_tokens.float(), reduction="mean")
            LDC_loss = F.mse_loss(noise_pred.float(), tar_noise_pred.float(), reduction="mean")

            loss = (LDC_loss+DFM_loss)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            DFM_loss_sum += DFM_loss.item()
            LDC_loss_sum += LDC_loss.item()
            if (step+1)%save_steps == 0:
                if accelerator.is_main_process:
                    print("Epoch {}, step {}, DFM_loss_sum: {}, LDC_loss_sum: {}".format(
                        epoch, step, DFM_loss_sum.item(), LDC_loss_sum.item()))
                    DFM_loss_sum = 0
                    LDC_loss_sum = 0
                torch.save(imr.state_dict(), f'./models/save_model/IMR_{epoch+1}_{step+1}.bin')
                torch.save(keypoint_encoder.state_dict(), f'./models/save_model/keypoint_encoder_{epoch+1}_{step+1}.bin')


        
        
        
        
            

                     
if __name__ == "__main__":
    main()    

