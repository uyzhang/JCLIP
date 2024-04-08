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
