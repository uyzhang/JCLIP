import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='A')

args = parser.parse_args()

model, preprocess = clip.load("ViT-B-32.pkl")
classes = open('Dataset/classes.txt').read().splitlines()

# remove the prefix Animal, Thu-dog, Caltech-101, Food-101

new_classes = []
for c in classes:
    c = c.split(' ')[0]
    if c.startswith('Animal'):
        c = c[7:]
    if c.startswith('Thu-dog'):
        c = c[8:]
    if c.startswith('Caltech-101'):
        c = c[12:]
    if c.startswith('Food-101'):
        c = c[9:]
    c = 'a photo of ' + c
    new_classes.append(c)

text = clip.tokenize(new_classes)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)

split = 'TestSet' + args.split

imgs_dir = 'Dataset/' + split
imgs = os.listdir(imgs_dir)

save_file = open('result.txt', 'w')

preds = []
with jt.no_grad():
    for img in tqdm(imgs):
        img_path = os.path.join(imgs_dir, img)
        image = Image.open(img_path)
        image = preprocess(image).unsqueeze(0)
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 *
                      image_features @ text_features.transpose(0, 1)).softmax(
                          dim=-1)
        # top5 predictions
        _, top_labels = text_probs[0].topk(5)
        preds.append(top_labels)
        # save top5 predictions to file
        save_file.write(img + ' ' +
                        ' '.join([str(p.item()) for p in top_labels]) + '\n')
