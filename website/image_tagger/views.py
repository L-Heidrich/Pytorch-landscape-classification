from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.shortcuts import render
from django.urls import reverse
from django.views.generic import TemplateView
from django.views.decorators.csrf import csrf_protect, csrf_exempt
import torchvision.transforms as transforms
from PIL import Image
import torch
from image_tagger.forms import ImageForm
from .models import ImageModel


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
        form = ImageForm(request.POST, request.FILES)
        print("valid")
        # form.save()
        my_file = request.FILES.get("file")
        img = Image.open(my_file)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.to('cuda')

        preds = model(img)
        _, preds = torch.max(preds, dim=1)
        result = pred_translator[preds.item()]
        results_global.append(result)
        request.session['result'] = results_global
        # ImageModel.objects.create(img=my_file, name=result)
        print(result)
    else:
        request.session.flush()
        results_global.clear()
    return render(request, 'home.html', {})


def index(request):
    res = request.session.get("result")
    request.session.flush()
    results_global.clear()
    return render(request, 'upload.html', {"result": res})
