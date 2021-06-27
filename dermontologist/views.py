from django.shortcuts import render, HttpResponse
from . import function as f 
from django.http import JsonResponse


def predict(request):
    if request.method == 'GET':
        return render(request, 'index.html')
    elif request.method == 'POST':
        # we will get the file from the request
        file = request.FILES.get('image')
        # convert that to bytes
        img_bytes = file.read()
        class_id, class_name =  f.get_prediction(image_bytes=img_bytes)
        print(class_id)
        return render(request, 'index.html', {'response':class_name})