from django import forms
from image_tagger.models import ImageModel


class ImageForm(forms.ModelForm):
    class Meta:
        model = ImageModel
        fields = ("name", "img")
