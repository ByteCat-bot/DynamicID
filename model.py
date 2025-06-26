import torch.nn as nn
import math
import torch



class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class KeyPointEncoder(nn.Module):
    def __init__(self, embedding_dim=768):
        super(KeyPointEncoder, self).__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(Conv2d(1, 16, kernel_size=7, stride=1, padding=3)), 

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),     
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),   
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),
            
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),     
            Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)),])
        
    def forward(self, landmark):
        for f in self.blocks:
            landmark = f(landmark)
            
        return landmark
    
    



def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
    
    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs*heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, latents, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)



 
    
    

class MultAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)


        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        b, l, _ = x.shape

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)







class IMR(nn.Module):
    def __init__(
        self,
        dim=1024,
        dim_head=64,
        heads=16,
        num_queries=4,
        embedding_dim=768,
        txt_dim = 768,
        ff_mult=4,
        erase_layer_num=4,
        drive_layer_num=4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.dim = dim

        
        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(embedding_dim+txt_dim),
            nn.Linear(embedding_dim+txt_dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_queries*dim),
            nn.GELU(),
        )
        

        
        
        
        self.norm_in = nn.LayerNorm(embedding_dim)
        self.proj_in = nn.Linear(embedding_dim, dim)


        
        self.proj_out = nn.Linear(dim ,embedding_dim)
        self.norm_out = nn.LayerNorm(dim)

        self.disentangleNet = nn.ModuleList([])
        for _ in range(erase_layer_num):
            self.disentangleNet.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        MultAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        

        self.entangleNet = nn.ModuleList([])
        for _ in range(drive_layer_num):
            self.entangleNet.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        MultAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        



    def forward(self, x, src_landmark, tgt_landmark, src_txt_embed, tgt_txt_embed):
        src_feature = torch.cat([src_landmark, src_txt_embed], dim=1)
        tgt_feature = torch.cat([tgt_landmark, tgt_txt_embed], dim=1)
        
        all_feature = torch.cat([src_feature, tgt_feature], dim=0)
        latents = self.feature_fusion(all_feature)
        latents = latents.reshape(latents.shape[0], self.num_queries , self.dim)
        
        
        src_latents = latents[:1]
        tgt_latents = latents[1:]
        
        x = self.norm_in(x)
        x = self.proj_in(x)

        for attn1, attn2, ff in self.disentangleNet:
            x = attn1(x, src_latents) + x 
            x = attn2(x) + x 
            x = ff(x) + x
            

        x = x.repeat(tgt_latents.shape[0], 1, 1)
        
            
        for attn1, attn2, ff in self.entangleNet:
            x = attn1(x, tgt_latents) + x
            x = attn2(x) + x 
            x = ff(x) + x
            


        x = self.norm_out(x)
        x  = self.proj_out(x)
        return x



