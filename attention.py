import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.lora import LoRALinearLayer
import math



class SAAProcessor(nn.Module):

    def __init__(
            self, 
            hidden_size, 
            cross_attention_dim=None, 
            rank=4, 
            network_alpha=None, 
            lora_scale=1.0, 
            scale=1.0, 
            num_tokens=4,
        ):
        super().__init__()
        self.rank = rank
        self.lora_scale = lora_scale
        self.num_tokens = num_tokens

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        
        
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.id_scale = scale

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        
        self.id_num = 1,
        self.record_map = False
        self.activate_query = True
        self.bbox = None
        self.attn_maps = None
        self.root_img_width = None
        self.root_img_height = None
        self.count = 0

            
    
    
    def compute_activate_weight(self, score, bboxs):
        bsz, num_heads, query_len, _ = score.shape
        if self.root_img_width is None or self.root_img_height is None:
            width = height = int(math.sqrt(query_len))
        else:
            ratio = int(math.sqrt(self.root_img_width*self.root_img_height//query_len))
            width,height = self.root_img_width//ratio, self.root_img_height//ratio
            
        weight = score.permute(0,2,1,3)
        weight = weight.reshape(bsz, query_len, -1).sum(dim=-1)
        min_vals = weight.min(dim=1, keepdim=True).values
        max_vals = weight.max(dim=1, keepdim=True).values
        activate_weight = (weight - min_vals) / (max_vals - min_vals)
        if self.count < self.infer_end:
            infer_scale = self.infer_scale
        else:
            infer_scale = 0
        self.count+=1
        
        
        activate_weight[activate_weight < 0.3] = 0
        ## exp
        if query_len == 9216:
            self.attn_maps.append(activate_weight)
        
        activate_weight = activate_weight.reshape(bsz, height, width)/(infer_scale+1)
        
            
        score = score.reshape(bsz,num_heads,height,width,-1)
        for idx,bbox in enumerate(bboxs):
            y_min, x_min, y_max, x_max = bbox
            y_min = int(y_min * height)
            x_min = int(x_min * width)
            y_max = int(y_max * height)
            x_max = int(x_max * width)
            activate_weight[:,y_min:y_max, x_min:x_max] = activate_weight[:,y_min:y_max, x_min:x_max]*(infer_scale+1) + infer_scale
            for i in range(len(bboxs)):
                if i != idx:
                    score[:,:,y_min:y_max, x_min:x_max,i*4:(i+1)*4] = float('-inf')
        activate_weight = activate_weight.reshape(bsz, query_len)
        score = score.reshape(bsz, num_heads, query_len, -1)
        
        
        if len(bboxs) == 1:
            token_num = score.shape[3]
            if token_num/4 >1 :
                score[:,:,:,:4] = score[:,:,:,:4] * self.mix_scale
                score[:,:,:,4:] = score[:,:,:,4:] * (1-self.mix_scale)
        
        
        return score, activate_weight
        
    

        

    def __call__(
        self, 
        attn,
        hidden_states, 
        encoder_hidden_states=None, 
        attention_mask=None, 
        scale=1.0,
        temb=None,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + self.lora_scale * self.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens*self.id_num
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + self.lora_scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + self.lora_scale * self.to_v_lora(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        
        scale = 1 / math.sqrt(head_dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)

        hidden_states = torch.matmul(attn_weights, value)
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)
        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        

        
        scale = 1 / math.sqrt(math.sqrt(head_dim))
        score = (query * scale) @ (ip_key * scale).transpose(-2, -1)
        if self.activate_query:
            mask_score, activate_weight = self.compute_activate_weight(score, self.bbox)
        else:
            mask_score = score
            activate_weight = torch.ones((batch_size, query.shape[2])).to(query.device,query.dtype)
        activate_weight = activate_weight.unsqueeze(-1)
        
        attn_weight = torch.softmax(mask_score.float(), dim=-1).type(score.dtype)
        ip_hidden_states = attn_weight @ ip_value
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        ip_hidden_states = activate_weight * ip_hidden_states
            
           
        hidden_states = hidden_states + self.id_scale * ip_hidden_states
        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + self.lora_scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states










      
class SAProcessor(nn.Module):
    
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        rank=4,
        network_alpha=None,
        lora_scale=1.0,
    ):
        super().__init__()
        
        self.rank = rank
        self.lora_scale = lora_scale
        
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + self.lora_scale * self.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + self.lora_scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + self.lora_scale * self.to_v_lora(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)


        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + self.lora_scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    

