from typing import Tuple, Union

import numpy as np
import jittor as jt
from jittor import nn, init
from .mha import MultiheadAttention


def normal_(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        init.gauss_(module.weight, mean, std)
    if hasattr(module, 'bias') and isinstance(
            module.bias, jt.Var) and module.bias is not None:
        init.constant_(module.bias, bias)


class LayerNorm(nn.LayerNorm):

    def execute(self, x):
        ret = super().execute(x)
        return ret


class QuickGELU(nn.Module):

    def execute(self, x):
        return x * jt.sigmoid(1.702 * x)


class MLP(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_model * 4)
        self.gelu = QuickGELU()
        self.c_proj = nn.Linear(d_model * 4, d_model)

    def execute(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model, n_head, attn_mask):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = MLP(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False,
                         attn_mask=self.attn_mask)[0]

    def execute(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def execute(self, x):
        return self.resblocks(x)


class VisionTransformer(nn.Module):

    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        scale = width**-0.5
        self.class_embedding = scale * jt.randn((width))
        self.positional_embedding = scale * jt.randn(
            ((input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = scale * jt.randn((width, output_dim))

    def execute(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = jt.concat([
            self.class_embedding.to(x.dtype) + jt.zeros(
                (x.shape[0], 1, x.shape[-1]), dtype=x.dtype), x
        ],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            # vision
            image_resolution: int,
            vision_layers: Union[Tuple[int, int, int, int], int],
            vision_width: int,
            vision_patch_size: int,
            # text
            context_length: int,
            vocab_size: int,
            transformer_width: int,
            transformer_heads: int,
            transformer_layers: int):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(input_resolution=image_resolution,
                                        patch_size=vision_patch_size,
                                        width=vision_width,
                                        layers=vision_layers,
                                        heads=vision_heads,
                                        output_dim=embed_dim)

        self.transformer = Transformer(width=transformer_width,
                                       layers=transformer_layers,
                                       heads=transformer_heads,
                                       attn_mask=self.build_attention_mask())

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = jt.empty(
            (self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = jt.empty((transformer_width, embed_dim))
        self.logit_scale = jt.ones([]) * np.log(1 / 0.07)

        self.initialize_parameters()

    def initialize_parameters(self):
        normal_(self.token_embedding.weight, std=0.02)
        normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers)**-0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width)**-0.5
        for block in self.transformer.resblocks:
            normal_(block.attn.in_proj_weight, std=attn_std)
            normal_(block.attn.out_proj.weight, std=proj_std)
            normal_(block.mlp.c_fc.weight, std=fc_std)
            normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        mask = jt.empty((self.context_length, self.context_length))
        mask.fill_(float("-inf"))
        mask = jt.triu(mask, 1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[jt.arange(x.shape[0]),
              text.argmax(dim=-1)[0]] @ self.text_projection
        return x

    def execute(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1)**0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2] for k in state_dict
                    if k.startswith(f"visual.layer{b}")))
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] -
             1)**0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == state_dict[
            "visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2] for k in state_dict
            if k.startswith("transformer.resblocks")))

    model = CLIP(embed_dim, image_resolution, vision_layers, vision_width,
                 vision_patch_size, context_length, vocab_size,
                 transformer_width, transformer_heads, transformer_layers)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    model.load_parameters(state_dict)
    return model.eval()
