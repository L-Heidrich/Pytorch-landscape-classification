from django.urls import path

from . import views


urlpatterns = [
    path('home/', views.upload_view),
    path('', views.upload_view),
    path('upload/', views.index, name='upload'),
]