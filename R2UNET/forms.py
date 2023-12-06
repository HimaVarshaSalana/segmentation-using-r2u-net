from django import forms

from PIL import Image

class ImageInputForm(forms.Form):
    image = forms.ImageField()
