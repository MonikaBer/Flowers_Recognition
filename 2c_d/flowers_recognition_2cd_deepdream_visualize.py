import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm
import scipy.ndimage as nd

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

def deprocess(image_np):
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image_np = np.clip(image_np, 0.0, 255.0)
    return image_np

def clip(image_tensor):
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
    return image_tensor

def dream(image, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor
    image = Variable(Tensor(image), requires_grad=True)
    for i in range(iterations):
        model.zero_grad()
        out = model(image)
        loss = out.norm()
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()


def deep_dream(image, model, iterations, lr, octave_scale, num_octaves):
    """ Main deep dream method """
    image = preprocess(image).unsqueeze(0).cpu().data.numpy()

    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return deprocess(dreamed_image)


if __name__ == "__main__":
    image_path = "images/daisy.jpg"
    #image_path = "images/dandelion.jpg"
    #image_path = "images/rose.jpg"
    #image_path = "images/sunflower.jpg"
    #image_path = "images/tulip.jpg"
    iterations = 30
    at_layer = 16
    lr = 0.02
    octave_scale = 1.4
    num_octaves = 6

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the model
    # network = models.resnet50(pretrained=True)
    network = torch.load('./drive/My Drive/2cd/2c_trained_model.pt')
    # network = torch.load('./drive/My Drive/2cd/2d_trained_model.pt')
    layers = list(network.children())[0:4] + \
            list(network.layer1) + \
            list(network.layer2) + \
            list(network.layer3) + \
            list(network.layer4) + \
            list(network.children())[8:10]
    model = nn.Sequential(*layers[: (at_layer + 1)])
    model.eval()
    if torch.cuda.is_available:
        model = model.cuda()

    # Extract deep dream image
    dreamed_image = deep_dream(
        image,
        model,
        iterations,
        lr,
        octave_scale,
        num_octaves
    )

    dreamed_image = np.true_divide(dreamed_image, np.amax(dreamed_image))
    plt.figure(figsize=(20, 20))
    plt.imshow(dreamed_image)
    plt.show()