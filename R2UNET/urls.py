from django.urls import path
from django.urls import re_path
from . import views

urlpatterns = [
    path('', views.home, name='app-home'),
    #path('about/',views.about,name='app-about'),
    path('form/', views.form, name = 'app-form'),
    path('result/',views.result,name ='result')
    
]