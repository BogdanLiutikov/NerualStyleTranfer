import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128


def tensor_toImage(tensor: torch.Tensor) -> Image:
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


cnn = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
content_layers_default = ['relu5_2']
style_layers_default = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               content_img, style_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    block = 1
    i = 1
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            name = f'conv{block}_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu{block}_{i}'
            layer = nn.ReLU(inplace=False)
            i += 1
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool{block}_{i}'
            block += 1
            i = 1
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn{block}_{i}'
        else:
            raise RuntimeError(f'Неизвестный слой: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss{block}_{i}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss{block}_{i}', style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000, content_weight=1,
                       socketio=None):
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std,
                                                                     content_img=content_img,
                                                                     style_img=style_img)
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            socketio.emit('update_iterations', {
                'content_loss': content_score.item(),
                'style_loss': style_score.item(),
                'loss': loss.item(),
                'iterations': run[0]
            })
            socketio.sleep(0)

            run[0] += 1

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def main(content_img, style_img, *, socketio=None, iterations=100, style_weight=1e6, content_weight=1):
    content_image_size = content_img.size
    scale_factor = imsize / max(content_image_size)
    resize = (int(content_image_size[0] * scale_factor), int(content_image_size[1] * scale_factor))
    resize = resize[::-1]
    preprocess_resize_toTensor = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()])
    content_img_tensor = preprocess_resize_toTensor(content_img).unsqueeze(0).to(device, torch.float)
    style_img_tensor = preprocess_resize_toTensor(style_img).unsqueeze(0).to(device, torch.float)
    input_img = content_img_tensor.clone() + (torch.rand(content_img_tensor.size()).to(device) - 0.5) / 30
    result = run_style_transfer(
        cnn, cnn_normalization_mean, cnn_normalization_std,
        content_img=content_img_tensor,
        style_img=style_img_tensor,
        input_img=input_img,
        num_steps=iterations,
        style_weight=style_weight,
        content_weight=content_weight,
        socketio=socketio
    )
    return tensor_toImage(result)
