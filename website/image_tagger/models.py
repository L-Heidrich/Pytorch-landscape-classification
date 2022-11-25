from django.db import models

# Create your models here.
from django.db import models


class ImageModel(models.Model):
    img = models.ImageField(upload_to='images/', default= None)
    name = models.CharField(max_length=50, default=None)
