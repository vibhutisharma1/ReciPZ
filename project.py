
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import PIL


import torch
from torch import nn
from torchvision.models import resnet
from torchvision import transforms
from flask import jsonify


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

app = Flask(__name__)
MODEL_PATH = 'models/model.th'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
                                transforms.Resize((224,224)),
                                #transforms.CenterCrop((224,224)),
                                transforms.ToTensor(),transforms.Normalize(torch.Tensor(mean),
                                                        torch.Tensor(std))])
                                                    

# Load your trained model
def load_network(net_model, net_name, dropout_ratio, class_names):
    for name, param in net_model.named_parameters():
        param.requires_grad = False

    if net_name.startswith('resnet'):
        num_ftrs = net_model.fc.in_features
        net_model.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
                                     nn.ReLU(),
                                     nn.Dropout(p=dropout_ratio),
                                     nn.Linear(256, len(class_names)))


    
    total_params = sum(param.numel() for param in net_model.parameters())
    print(f'{total_params:,} total parameters')

    total_trainable_params = sum(param.numel() for param in net_model.parameters() if param.requires_grad)
    print(f'{total_trainable_params:,} training parameters')
    
    return net_model

net_model = resnet.resnet18(pretrained=False)
net_name = 'resnet18'

dropout_ratio = 0.25
class_names = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 
'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 
'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 
'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 
'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

net_model = load_network(net_model, net_name, dropout_ratio, class_names)

#capsicum recipes


net_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
net_model.eval() # IMPORTANT


print('Model loaded. Check http://127.0.0.1:5000/')



def model_predict(file, model):
    img = PIL.Image.open(file)
    x = test_transform(img)
    x = x.unsqueeze(0)
    preds = model.forward(x)
    index = torch.argmax(preds,1).item()
    
    return class_names[index]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # img = PIL.Image.open(f)
        # print(img)
        # print(test_transform(img))

        # Save the file to ./uploads

        # Make prediction
        result = model_predict(f, net_model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)


# class Recipe:
#   def __init__(title, image_url, recipe_link, info):
#     self.title = title
#     self.image_url = image_url
#     self.recipe_link = recipe_link
#     self.info = info

# tomato_soup = Recipe("Tomato Soup", https://www.inspiredtaste.net/wp-content/uploads/2016/08/Tomato-Soup-Recipe-2-1200-768x512.jpg ,https://www.inspiredtaste.net/27956/easy-tomato-soup-recipe/36, "Quick, Velvety, and Simple Tomato Soup")

# #chicken_curry = Recipe("",https://txconvergent.slack.com/archives/C01RJGHQTQF/p1619832169010800 , ,"Decadent and Delicious .." )

# @app.route('/_get_current_user')
# def get_current_user():
#     return jsonify(username=g.user.username,
#                    email=g.user.email,
#                    id=g.user.id)




