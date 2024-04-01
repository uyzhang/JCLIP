# JCLIP

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://arxiv.org/abs/2103.00020)


JCLIP为CLIP的Jittor版本，CLIP（Contrastive Language-Image Pre-Training）是一个在各种（图像、文本）对上训练的神经网络。可以用自然语言指示它在给定图像的情况下预测最相关的文本片段，而无需直接对任务进行优化，这与 GPT-2和3的zero-shot功能类似。


## 网络结构

![CLIP](CLIP.png)


## 使用方法

### 安装依赖环境
```bash
pip install jittor
pip install ftfy regex tqdm
python setup.py develop
```

### 模型权重

下载[VIT-B-32](https://github.com/uyzhang/JCLIP/releases/tag/%E6%9D%83%E9%87%8D)或利用转换脚本，将PyTorch权重转换为Jittor权重。

```python
import torch
import jittor as jt
clip = torch.load('ViT-B-32.pt').state_dict()

for k in clip.keys():
    clip[k] = clip[k].float().cpu()
jt.save(clip, 'ViT-B-32.pkl')
```


### demo
```python
import jittor as jt
import jclip as clip
from PIL import Image

jt.flags.use_cuda = 1

model, preprocess = clip.load("ViT-B-32.pkl")

image = preprocess(Image.open("CLIP.png")).unsqueeze(0)

text = clip.tokenize(["a diagram", "a dog", "a cat"])

with jt.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
```


## 第四届计图比赛Baseline
- Training-Free版本
```bash
python baseline.py
```
- Training版本
```bash
python baseline_ft.py
```

得到result.txt，打包为zip，提交即可。
