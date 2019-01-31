from PIL import Image
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

def process_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
        Process an image, given the image path, to be fully prepared to 
        run through the ResNet 152 model. 
    '''

    if type(image) is str:
        image = Image.open(image)

    if image.size[0] > image.size[1]:
        new_height = 256
        new_width = int((new_height / image.size[1]) * image.size[0])
    else:
        new_width = 256
        new_height = int((new_width / image.size[0]) * image.size[1])

    image.thumbnail((new_width, new_height))

    # Images should be 224 x 224

    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2

    image = image.crop((left, top, right, bottom))

    assert image.size == (224, 224), "Processed image does not meet the size requirement: (224, 224)"    

    # Normalization
    image = np.array(image)
    image = image / 255
    image = (image - mean) / std

    image = image.transpose((2, 0, 1))

    return image

def topk_prediction(image, model, topk=5):
    '''
        Prediction of the top K most probable classes of a given image
    '''

    assert type(image) is str or torch.Tensor, "Image does not meet the type requirement: must be either string or tensor"

    if type(image) is str:
        image = process_image(image)

        tensor_image = torch.from_numpy(image)
        tensor_image = tensor_image.float()
    else:
        tensor_image = image.clone()

    if len(tensor_image.size()) == 3:
        tensor_image.unsqueeze_(0)
    
    # Prediction
    output = model(tensor_image)
    softmax = nn.Softmax(dim=1)
    output = softmax(output)

    # Getting topk prediction
    probabilities, classes = output.topk(topk, dim=1)

    probabilities = probabilities.squeeze().detach().numpy()
    classes = classes.squeeze().detach().numpy()
    classes = [str(c) for c in classes]

    return probabilities, classes


def imshow(image, ax=None, title=None):
    '''
        Image show for a Tensor
    '''

    if ax is None:
        _, ax = plt.subplots(figsize=(12.0, 10.0))

    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image = std * image + mean

    image = np.clip(image, 0, 1)

    ax.axis('off')
    ax.imshow(image)
    ax.grid(False)
    ax.set_title(title)

    return ax

def sanity_check(image, label, model):
    '''
        Sanity check to assert if the model is predicting the labels correctly.
    '''
    assert type(label) is str or torch.Tensor, 'Label must be either a string or a tensor'
  
    if type(label) is torch.Tensor:
        label = str(label.item())
        
    class_idx = model.to_idx[label]
    class_name = model.idx_to_class[class_idx]
    
    # Making the predictions
    probabilities, classes = topk_prediction(image, model, topk=5)
    
    # Getting the name of the most probable classes
    idxs = [model.to_idx[c] for c in classes]
    classes_name = [model.idx_to_class[str(i)] for i in idxs]
    
    # Plotting the figure
    fig, (ax1, ax2) = plt.subplots(figsize=(9, 12), ncols=2)
    imshow(image, ax=ax1, title=class_name)
    
    ax2.barh(np.arange(5), probabilities)
    ax2.set_aspect(0.2)
    ax2.set_yticks(np.arange(5))
    
    ax2.set_yticklabels(classes_name, size='medium')
    
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    
    plt.tight_layout()

def predict(image, model):
    '''
        Predicting image class
    '''
    assert type(image) is str or torch.Tensor()

    model.eval()

    if type(image) is str: 
        image = process_image(image)

        tensor_image = torch.from_numpy(image)
        tensor_image = tensor_image.float()
    else:
        tensor_image = tensor_image.clone()

    if len(tensor_image.size()) == 3:
        tensor_image.unsqueeze_(0)

    output = model(tensor_image)
    softmax = nn.Softmax(dim=1)
    output = softmax(output)

    _, predicted_class = output.topk(1, dim=1)

    return predicted_class
