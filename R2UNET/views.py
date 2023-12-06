import subprocess
import os
from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib import messages

from django.contrib.auth.models import User

from .forms  import ImageInputForm
from django.contrib.auth.decorators import login_required

from PIL import Image

#from utils.testing import first

# Create your views here.

def home(request):
    return render(request,'R2UNET/home.html')

@login_required
def form(request):
    if request.method == 'POST':
        form = ImageInputForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data.get('image')
            with open(os.path.join('media','test_image.tif'), 'wb') as file:
                for chunk in image.chunks():
                    file.write(chunk)
            image = Image.open('media/test_image.tif')
            image.save('media/test_image.png', "PNG")
            script_path = 'utils\\testing.py'
            subprocess.run(['python', script_path]) 
            return redirect('result')
    else:
        form = ImageInputForm()
    context = {
        'form' : form
    }
    return render(request,'R2UNET/form.html',context)

@login_required
def result(request):
    return render(request, 'R2UNET/result.html')