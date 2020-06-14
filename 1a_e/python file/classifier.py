# -*- coding: utf-8 -*-
"""snr.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13OmmeNFumsPdxlS8-d9uX39ryeF3mpRG
"""

!pip3 install torch==1.5.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

from google.colab import drive
drive.mount('/content/gdrive')

root_path = 'gdrive/My Drive/flowers'  #change dir to your project folder

# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

# Commented out IPython magic to ensure Python compatibility.
import copy
import os
import time
import torch
import torchvision.models as models
from torch import nn, device, optim
from torchvision import datasets
from torchvision.transforms import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
import math
# %matplotlib inline

data_dir = root_path
num_classes = 5
batch_size = 5
num_epochs = 15
# In feature extraction, we start with a pretrained model and only update the final layer weights from which we derive
# predictions. 
# It is called feature extraction because we use the pretrained CNN as a fixed feature-extractor, and only change 
# the output layer.
feature_extract = True

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def quadratic(input, weight, bias=None):
    # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
    tens_ops = (input, weight)
    if not torch.jit.is_scripting():
        if any([type(t) is not torch.Tensor for t in tens_ops]) and nn.functional.has_torch_function(tens_ops):
            return nn.functional.handle_torch_function(linear, tens_ops, input, weight, bias=bias)
    if input.dim() == 2 and bias is not None:
        # fused op is marginally faster

        ret = torch.addmm(bias, input, weight.t())
        #out = nn.functional.normalize(torch.exp(ret))
        out = nn.functional.normalize(torch.pow(ret, 2))
        norm = ((2*out) -1 )* 10
        
        # weights = torch.transpose(weight.t(), 0, 1)
        
        # sub = torch.sub(input, weights)
        # #norm = torch.norm(sub, dim=1)
        # #matrix_norm = torch.unsqueeze(norm,1).repeat(1,num_classes)
        # #print(matrix_norm)
        # gamma = - 1/2
        # arg = torch.mul(sub, gamma)
        
        # exp = torch.exp(arg[:, :num_classes])
        # out = nn.functional.normalize(exp, dim=0)
        # norm = ((2*out) -1 )* 10
        # #norm = torch.transpose(norm, 0, 1)
        ret = norm
    else:
        print("else branch")
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
    return ret

class Quadratic(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Quadratic, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return quadratic(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

def initialize_model(num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    torch.manual_seed(10)
    model_ft = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = Quadratic(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train_model(model, dataloaders, optimizer, criterion, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs) 
                        labels_2d = torch.unsqueeze(labels,1).repeat(1,num_classes)
                        loss = criterion(outputs, labels_2d)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.MultiLabelMarginLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict,optimizer_ft, criterion, num_epochs=num_epochs, is_inception=False)

torch.save(model_ft, 'svm_trained_model.pt')
#model_ft = torch.load('gdrive/My Drive/training outcome/trained_model.pt')
#model_ft = torch.load('gdrive/My Drive/training outcome/trained_model.pt', map_location=torch.device('cpu'))

input_size = 224

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
#model_ft = model_ft.to(device)

transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)

train_dataset = torch.utils.data.DataLoader(
        image_dataset, shuffle=False, num_workers=16
    )

model_ft.eval()
all_features = []

for data, target in train_dataset:
  out = model_ft(data)
  all_features.append(out.detach().numpy())

torch.save(torch.tensor(all_features), "val_features_tensor.pt")

features_dir = "gdrive/My Drive/training outcome/"
t = torch.load(features_dir +'all_features_tensor.pt').cpu()
X_train = t.detach().numpy().reshape((4170, 2048))

t = torch.load(features_dir +'val_features_tensor.pt').cpu()
X_test = t.detach().numpy().reshape((150, 2048))

print(X_train.shape)
print(X_test.shape)

daisy_dir = data_dir + '/train/daisy'
daisy_len= len([name for name in os.listdir(daisy_dir) if os.path.isfile(os.path.join(daisy_dir, name))])
print(daisy_len)
daisy_list = ['daisy'] * daisy_len

dandelion_dir = data_dir + '/train/dandelion'
dandelion_len= len([name for name in os.listdir(dandelion_dir) if os.path.isfile(os.path.join(dandelion_dir, name))])
print(dandelion_len)
dandelion_list = ['dandelion'] * dandelion_len

rose_dir = data_dir + '/train/rose'
rose_len= len([name for name in os.listdir(rose_dir) if os.path.isfile(os.path.join(rose_dir, name))])
print(rose_len)
rose_list = ['rose'] * rose_len

sunflower_dir = data_dir + '/train/sunflower'
sunflower_len= len([name for name in os.listdir(sunflower_dir) if os.path.isfile(os.path.join(sunflower_dir, name))])
print(sunflower_len)
sunflower_list = ['sunflower'] * sunflower_len

tulip_dir = data_dir + '/train/tulip'
tulip_len= len([name for name in os.listdir(tulip_dir) if os.path.isfile(os.path.join(tulip_dir, name))])
print(tulip_len)
tulip_list = ['tulip'] * tulip_len

y_train = np.concatenate([daisy_list, dandelion_list, rose_list, sunflower_list, tulip_list])

#print(y_train.shape)

daisy_dir = data_dir + '/val/daisy'
daisy_len= len([name for name in os.listdir(daisy_dir) if os.path.isfile(os.path.join(daisy_dir, name))])
#print(daisy_len)
daisy_list = ['daisy'] * daisy_len

dandelion_dir = data_dir + '/val/dandelion'
dandelion_len= len([name for name in os.listdir(dandelion_dir) if os.path.isfile(os.path.join(dandelion_dir, name))])
#print(dandelion_len)
dandelion_list = ['dandelion'] * dandelion_len

rose_dir = data_dir + '/val/rose'
rose_len= len([name for name in os.listdir(rose_dir) if os.path.isfile(os.path.join(rose_dir, name))])
#print(rose_len)
rose_list = ['rose'] * rose_len

sunflower_dir = data_dir + '/val/sunflower'
sunflower_len= len([name for name in os.listdir(sunflower_dir) if os.path.isfile(os.path.join(sunflower_dir, name))])
#print(sunflower_len)
sunflower_list = ['sunflower'] * sunflower_len

tulip_dir = data_dir + '/val/tulip'
tulip_len= len([name for name in os.listdir(tulip_dir) if os.path.isfile(os.path.join(tulip_dir, name))])
#print(tulip_len)
tulip_list = ['tulip'] * tulip_len

y_test = np.concatenate([daisy_list, dandelion_list, rose_list, sunflower_list, tulip_list])

print(y_test.shape)

#https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))