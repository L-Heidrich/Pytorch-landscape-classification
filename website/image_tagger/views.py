from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.shortcuts import render
from django.urls import reverse
from django.views.generic import TemplateView
from django.views.decorators.csrf import csrf_protect, csrf_exempt
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
from .models import Images


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

model = torch.jit.load("../Models/script.pt")
model.eval()
transform = transforms.Compose(
    [
        transforms.Resize((100, 100)),
        transforms.ToTensor(),

    ])

results_global = []


def upload_view(request):
    if request.method == "POST":
        # form = ImageForm(request.POST, request.FILES)

        my_file = request.FILES.get("file")
        img = Image.open(my_file)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.to('cuda')

        preds = model(img)
        preds = preds.cpu()
        preds = np.where(preds[0] > 1)

        results = [pred_translator[i] for i in preds[0]]
        Images.objects.create(img=my_file, tags=results)

    return render(request, 'home.html', {})


def index(request):
    images = Images.objects.all()
    return render(request, 'upload.html', {"result": images})
