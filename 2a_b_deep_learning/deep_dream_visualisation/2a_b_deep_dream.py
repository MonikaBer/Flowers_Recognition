import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
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
    model_ft, input_size = initialize_model(num_classes, use_pretrained=True)
    model_ft.load_state_dict(torch.load("/content/drive/My Drive/monika/2b_weights.h5"))

    # layers = list(model_ft.children())[0:4] + list(model_ft.layer1) + list(model_ft.layer2) + list(model_ft.layer3) + list(model_ft.layer4) + list(model_ft.children())[8:10]
    # model_ft = nn.Sequential(*layers[: (at_layer + 1)])

    model_ft.cuda()

    model_ft.eval()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input_image", type=str, default="/content/drive/My Drive/flowers/rest/5673728_71b8cb57eb.jpg", help="path to input image")
    # parser.add_argument("--iterations", default=20, help="number of gradient ascent steps per octave")
    # parser.add_argument("--at_layer", default=16, type=int, help="layer at which we modify image to maximize outputs")
    # parser.add_argument("--lr", default=0.02, help="learning rate")
    # parser.add_argument("--octave_scale", default=1.4, help="image scale between octaves")
    # parser.add_argument("--num_octaves", default=10, help="number of octaves")
    # args = parser.parse_args()

    # parameters
    input_image = "/content/drive/My Drive/flowers/rest/5673728_71b8cb57eb.jpg"
    image = Image.open(input_image)
    iterations = 20
    at_layer = 16
    lr = 0.02
    octave_scale = 1.4
    num_octaves = 10

    # Define the model
    # model_ft = models.resnet50(pretrained=True)
    # layers = list(model_ft.children())[0:4] +
    #         list(model_ft.layer1) +
    #         list(model_ft.layer2) +
    #         list(model_ft.layer3) +
    #         list(model_ft.layer4) +
    #         list(model_ft.children())[8:10]
    # model = nn.Sequential(*layers[: (args.at_layer + 1)])
    # model.eval()
    # if torch.cuda.is_available:
    #     model = model.cuda()

    # Extract deep dream image
    dreamed_image = deep_dream(
        image,
        model_ft,
        iterations,
        lr,
        octave_scale,
        num_octaves
    )
    # network = models.resnet50(pretrained=True)
    # layers = list(network.children())[0:4] +
    #         list(network.layer1) +
    #         list(network.layer2) +
    #         list(network.layer3) +
    #         list(network.layer4) +
    #         list(network.children())[8:10]
    # model = nn.Sequential(*layers[: (args.at_layer + 1)])
    # model.eval()
    # if torch.cuda.is_available:
    #     model = model.cuda()

    # # Extract deep dream image
    # dreamed_image = deep_dream(
    #     image,
    #     model,
    #     iterations=args.iterations,
    #     lr=args.lr,
    #     octave_scale=args.octave_scale,
    #     num_octaves=args.num_octaves
    # )

    dreamed_image = np.true_divide(dreamed_image, np.amax(dreamed_image))
    # Save and plot image
    # os.makedirs("outputs", exist_ok=True)
    # filename = args.input_image.split("/")[-1]
    plt.figure(figsize=(20, 20))
    plt.imshow(dreamed_image)
    plt.imsave("/content/drive/My Drive/monika/2b_dreamed_5673728_71b8cb57eb.jpg", dreamed_image)
    # plt.show()ls