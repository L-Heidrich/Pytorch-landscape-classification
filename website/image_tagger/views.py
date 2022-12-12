from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.views.generic import TemplateView
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
from .models import Images
import os
from django.conf import settings
from torchvision.models import resnet18
import torch.nn as nn
from requests import session

class MainView(TemplateView):
    template_name = "home.html"


pred_translator = {
    0: "buildings",
    1: "forest",
    2: "glacier",
    3: "mountain",
    4: "sea",
    5: "street",
}

path = settings.STATIC_ROOT[0]
print(path)
model = resnet18()
model.fc = nn.Linear(in_features=512, out_features=6, bias=True)
model.load_state_dict(torch.load("staticfiles" + "\\resnet18.pth", map_location=torch.device('cpu')))
# model = torch.load(model_path)
model = model.cpu()
model.eval()
transform = transforms.Compose(
    [
        transforms.Resize((100, 100)),
        transforms.ToTensor(),

    ])

submitted_images = []
session_data = {}
results = []


def upload_view(request):
    if request.method == "POST":
        # form = ImageForm(request.POST, request.FILES)

        my_file = request.FILES.get("file")
        img = Image.open(my_file)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        #img = img.to('cuda')

        preds = model(img)
        preds = preds.cpu()
        preds = np.where(preds[0] > 1)
        results = [pred_translator[i] for i in preds[0]]

        Images.objects.create(img=my_file, tags=results)

    return render(request, 'home.html', {})


def index(request):
    images = Images.objects.all()
    #images = session_data[request.session.session_key]

    return render(request, 'upload.html', {"result": images})
