import io
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)




from torchvision import models

# Make sure to pass `pretrained` as `True` to use the pretrained weights:
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.require_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 3))
model.load_state_dict(torch.load("modelRESNET50-5batch.pt"))

# Since we are using our model only for inference, switch to `eval` mode:

model.eval()


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return y_hat


import json

imagenet_class_index = json.load(open('./imagenet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

