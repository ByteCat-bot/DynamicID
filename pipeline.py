from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np 
import torch
from insightface.app import FaceAnalysis 
from safetensors import safe_open
from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.utils import _get_model_file
from attention import SAAProcessor, SAProcessor
from functions import ProjPlusModel
import cv2
from PIL import Image
from model import IMR, KeyPointEncoder
from torchvision import transforms
from ResEmoteNet import ResEmoteNet
import torch.nn.functional as F

### Download the pretrained model from huggingface and put it locally, then place the model in a local directory and specify the directory location.
class DynamicIDStableDiffusionPipeline(StableDiffusionPipeline):
    
        
    
    @validate_hf_hub_args
    def load_DynamicID(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        SAA_path: str,
        IMR_path = None,
        IMR_depth = 1,
        subfolder: str = '',
        image_encoder_path: str = './models/laion--CLIP-ViT-H-14-laion2B-s32B-b79K',  
        torch_dtype = torch.float16,
        num_tokens = 4,
        lora_rank= 128,
        **kwargs,
    ):
        self.lora_rank = lora_rank 
        self.torch_dtype = torch_dtype
        self.num_tokens = num_tokens
        self.set_ip_adapter()
        self.image_encoder_path = image_encoder_path
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=self.torch_dtype
        )   
        self.clip_image_processor = CLIPImageProcessor()
        self.id_image_processor = CLIPImageProcessor()
        self.crop_size = 512
        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(512, 512))


        self.image_proj_model = ProjPlusModel(
            cross_attention_dim=self.unet.config.cross_attention_dim, 
            id_embeddings_dim=512,
            clip_embeddings_dim=self.image_encoder.config.hidden_size, 
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=self.torch_dtype)


        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=SAA_path,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict
    
        self.image_proj_model.load_state_dict(state_dict["proj_module"], strict=True)
        ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["SAA_module"], strict=True)
        
        if IMR_path is not None:     
            self.IMR = IMR(erase_layer_num=IMR_depth, drive_layer_num=IMR_depth).to(self.device,dtype=torch.float16)
            self.key_point_encoder = KeyPointEncoder().to(self.device,dtype=torch.float16)
            self.IMR.load_state_dict(torch.load(IMR_path,map_location="cpu"))
            key_point_encoder_path = IMR_path.replace('IMR','key_point_encoder')
            self.key_point_encoder.load_state_dict(torch.load(key_point_encoder_path,map_location="cpu"))
            self.landmark_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
            self.key_point_encoder.eval()
            print(f"Successfully loaded weights from checkpoint")
            
            # for Automated construction of face prompt
            self.emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

            self.emote_model = ResEmoteNet().to(self.device)
            checkpoint = torch.load('./models/ResEmoteNet/fer2013_model.pth', weights_only=True)
            self.emote_model.load_state_dict(checkpoint['model_state_dict'])
            self.emote_model.eval()
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.face_classifier = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )


    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 64 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    def set_ip_adapter(self):
        self.attn_maps = []
        unet = self.unet
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
                attn_procs[name] = SAProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=self.lora_rank
                ).to(self.device, dtype=self.torch_dtype)
            else:
                attn_procs[name] = SAAProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0, rank=self.lora_rank, num_tokens=self.num_tokens
                ).to(self.device, dtype=self.torch_dtype)
        
        unet.set_attn_processor(attn_procs)

    def get_src_face_prompt(self, image, angle):
        image = np.array(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        if len(faces)>0 :
            face = faces[0]
            (x, y, w, h) = face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            emotion_idx = self.get_max_emotion(x, y, w, h, image)
            emotion = self.emotions[emotion_idx]
        else:
            emotion = 'neutral'
        
        pose = ''
        yaw, pitch = angle
        
        if yaw>10:
            pose += 'facing up'
        elif yaw<-10:
            pose += 'facing down'
            
        if pose != '':
            pose += ', '
            
        if abs(pitch)<10:
            pose += 'in front view'
        else:
            pose += 'in side view'
        src_prompt = 'in a '+emotion+' expression, '+pose
        return src_prompt
 
    def detect_emotion(self, image):
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.emote_model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
        scores = probabilities.cpu().numpy().flatten()
        rounded_scores = [round(score, 2) for score in scores]
        return rounded_scores

    def get_max_emotion(self, x, y, w, h, image):
        crop_img = image[y : y + h, x : x + w]
        pil_crop_img = Image.fromarray(crop_img)
        rounded_scores = self.detect_emotion(pil_crop_img)    
        max_index = np.argmax(rounded_scores)
        return max_index

    def enable_processor(self, id_scale, activate_query, infer_scale, infer_end, record_map, bbox, id_num, attn_maps, width, height, mix_scale): 

        if bbox is None:
            bbox= []
            slice = 1/id_num
            for i in range(id_num):
                bbox.append([0,slice*i,1,slice*(i+1)])
                
        for attn in self.unet.attn_processors.values():
            if isinstance(attn, SAAProcessor):
                attn.id_scale = id_scale
                attn.attn_maps = attn_maps
                attn.activate_query = activate_query
                attn.record_map = record_map       
                attn.bbox = bbox
                attn.id_num = id_num
                attn.count = 0
                attn.infer_scale = infer_scale
                attn.infer_end = infer_end
                attn.mix_scale = mix_scale
                if width!=height:
                    attn.root_img_height = height
                    attn.root_img_width = width
                else:
                    attn.root_img_height = None
                    attn.root_img_width = None


    def get_image_embeds(self, faceid_embeds, face_image, s_scale, shortcut=False):

        clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.torch_dtype)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]
        
        faceid_embeds = faceid_embeds.to(self.device, dtype=self.torch_dtype)
        image_prompt_tokens = self.image_proj_model(faceid_embeds, clip_image_embeds, shortcut=shortcut, scale=s_scale)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(faceid_embeds), uncond_clip_image_embeds, shortcut=shortcut, scale=s_scale)
        return image_prompt_tokens, uncond_image_prompt_embeds


    def get_prepare_faceid(self, face_image):
        faceid_image = np.array(face_image)
        faces = self.app.get(faceid_image)
        if faces==[]:
            faceid_embeds = torch.zeros_like(torch.empty((1, 512)))
            yaw = 0
            pitch = 0
        else:
            faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
            yaw, pitch, _ = faces[0]['pose']
            ### TODO The prior extraction of FaceID is unstable and a stronger ID prior structure can be used.
            
        return faceid_embeds,(yaw,pitch)


    def cal_face_feature(self,input_id_image):
        faceid_embeds, angle = self.get_prepare_faceid(face_image=input_id_image)
        face_feature, uncond_image_prompt_embeds = self.get_image_embeds(faceid_embeds, face_image=input_id_image, s_scale=1.0, shortcut=False)
        self.uncond_image_prompt_embeds = uncond_image_prompt_embeds
        return face_feature, angle
    
    def get_landmark(self, face_image):
        face_image = face_image.resize((512,512))
        face_info = self.app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        if face_info:
            landmark = face_info[0]['landmark_2d_106']/(512/96)
            out_img = np.zeros([96, 96], dtype=np.uint8)
            for i in range(len(landmark)):
                if landmark[i][0] > 0 and landmark[i][1] > 0 and landmark[i][0] < 96 and landmark[i][1] < 96:
                    out_img[int(landmark[i][1]), int(landmark[i][0])] = 255
            out_img_pil = Image.fromarray(out_img.astype(np.uint8))
            return out_img_pil


        
    @torch.no_grad()
    def get_edited_token(self, src_img, tgt_face_prompt=None,tgt_landmark_path=None, src_face_prompt=None):
        src_token, angle = self.cal_face_feature(src_img).clone()
        if src_face_prompt is None:
            src_face_prompt = self.get_src_face_prompt(src_img, angle)
        
        src_landmark_img = self.get_landmark(src_img)
        src_landmark_img = src_landmark_img.convert('L')
        if tgt_landmark_path is not None:
            tgt_landmark_img = Image.open(tgt_landmark_path).convert('L')
        else:
            tgt_landmark_img = src_landmark_img
        src_landmark = self.landmark_transform(src_landmark_img).unsqueeze(0).to(self.device,dtype=torch.float16)
        tgt_landmark = self.landmark_transform(tgt_landmark_img).unsqueeze(0).to(self.device,dtype=torch.float16)
        landmark = torch.cat([src_landmark,tgt_landmark],dim=0)
        landmark_feature = self.key_point_encoder(landmark).reshape(2,768)
        src_lmk_ft = landmark_feature[:1]
        tgt_lmk_ft = landmark_feature[1:]
        face_prompts = [src_face_prompt,tgt_face_prompt]
        text_inputs = self.tokenizer(
                face_prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
        txt_embeds = self.text_encoder(text_inputs.input_ids.to(self.device))[1].detach().requires_grad_(False).to(torch.float16)
        src_txt_embeds = txt_embeds[:1]
        tgt_txt_embeds = txt_embeds[1:]
        edited_feature = self.IMR(src_token, src_lmk_ft, tgt_lmk_ft, src_txt_embeds, tgt_txt_embeds)
        return edited_feature
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        target_size: Optional[Tuple[int, int]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        face_features: Optional[torch.FloatTensor] = None,
        id_scale = 1,
        infer_end = 3,
        infer_scale = 4,
        activate_query: bool = False,
        record_map: bool = False,
        bbox=None,
        mix_scale = None,
        do_classifier_free_guidance = True
    ):
        
        self.attn_maps = []
        id_num = len(face_features[0])//self.num_tokens
        self.enable_processor(id_scale, activate_query, infer_scale, infer_end,record_map, bbox,id_num, self.attn_maps, width, height,mix_scale)
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )
        

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        
        
        
        prompt_embeds,negative_prompt_embeds = self.encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        

        # 5. Prepare the input ID images


        uncond_image_prompt_embeds = torch.cat([self.uncond_image_prompt_embeds]*id_num, dim=1)
        cross_attention_kwargs = {}

     

        negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
        prompt_embeds = torch.cat([prompt_embeds, face_features], dim=1)


        # 7. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 8. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
  
                

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 9.1 Post-processing
            image = self.decode_latents(latents)

            # 9.2 Run safety checker
            # image, has_nsfw_concept = self.run_safety_checker(
            #     image, device, prompt_embeds.dtype
            # )

            # 9.3 Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 9.1 Post-processing
            image = self.decode_latents(latents)

            # 9.2 Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=None
        )









