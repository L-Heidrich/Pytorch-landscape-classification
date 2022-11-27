from django.db import models

# Create your models here.
from django.db import models
from django.contrib.postgres.fields import ArrayField


class Images(models.Model):
    tags = ArrayField(models.CharField(max_length=50, default=None))
    img = models.ImageField(upload_to='images', default=None)
    name = models.ImageField(upload_to='images', default=None)