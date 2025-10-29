from django.shortcuts import render, HttpResponse,redirect
from django.contrib.auth.models import User,auth
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.conf import settings
from .ml import predict_disease_by_kushal,predict_disease_by_kushal2,evaluate_models,predict_using_image
from .extract_data import kushal_extract_doctors
from .models import PatientAppointment , Disease, Medicine
import json
from django.http import JsonResponse
import pandas as pd
import os

#
import requests
import pymongo

#training.csv ka file path
csv_file_path = os.path.join(settings.STATICFILES_DIRS[0], 'training.csv')

def index(request):
    if not request.user.is_authenticated:
        messages.info(request, "Login or Sign Up to use Doctor Functionality!")
    return render(request,"index.html")

def about(request):
    return render(request,'about.html')
def handleSignUp(request):
    if request.method=='POST':
        fname = request.POST['fname']
        lname = request.POST['lname']
        username = request.POST['username']
        password = request.POST['pass1']
        pass2 = request.POST['pass2']
        email = request.POST['email']

        if(User.objects.filter(username=username).exists()):
            messages.info(request,'Username taken !')
        elif(User.objects.filter(email=email).exists()):
            messages.info(request,'Email taken !')
        elif(password!=pass2):
            messages.info("Passwords dont match !")
        else:
            user = User.objects.create_user(username=username,password=password,email=email,first_name=fname,last_name = lname)
            user.save()
        return redirect('/')        

def handleLogin(request):
    if(request.method=='POST'):
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username=username,password=password)

        if user is not None:
            auth.login(request,user)
        else:
            messages.info(request," Wrong credentials ! Please enter your credentials properly !")
    return redirect('/')

def handleLogout(request):
    auth.logout(request)
    return redirect('/')

def predict(request):
    return render(request,'predict.html')


def book_doctor(request):
    if request.user.is_authenticated:
        # Check if the user has already booked an appointment
        if PatientAppointment.objects.filter(user=request.user).exists():
            messages.info(request, "You have already registered for an appointment! Please wait for further communication.")
        else:
            # Save the appointment
            PatientAppointment.objects.create(user=request.user)
            messages.info(request, "Booked Successfully! We will email you on your registered email by assigning our specialized doctor.")
        return redirect('/')
    else:
        messages.info(request, "You are not logged in! Please log in first!")
        return redirect('/')
    
def disease_predictor_by_symptoms(request):
    
    symptoms = pd.read_csv(csv_file_path, nrows=1).columns.tolist()
    if request.method == 'POST':
        selected_symptoms = request.POST.get('selected_symptoms', '[]')
        if len(selected_symptoms)==0:
            messages.warning(request," You have to select at least 1 symptom ! ")
            return render(request,"disease_predictor_by_symptoms.html",{'symptoms': symptoms})
        try:
            selected_symptoms_list = json.loads(selected_symptoms)  # Parse JSON string to a Python list
        except json.JSONDecodeError:
            selected_symptoms_list = []  # Default to an empty list if parsing fails 
        print(selected_symptoms_list)
        disease = predict_disease_by_kushal(selected_symptoms) 
        other_disease = predict_disease_by_kushal2(selected_symptoms)

        result = [disease]
        if other_disease['Naive Bayes'] not in result:
            result.append(other_disease['Naive Bayes'])
        if other_disease['K-Nearest Neighbors'] not in result:
            result.append(other_disease['K-Nearest Neighbors'])
        if other_disease['SVM'] not in result:
            result.append(other_disease['SVM'])

        


        print(disease)
        return render(request,"disease_prediction_result.html",{'result':result,'selected_symptoms':selected_symptoms_list})
        return HttpResponse("<h1> You are suffering from -> "+predict_disease_by_kushal(selected_symptoms) + "</h1>")
    
    return render(request,"disease_predictor_by_symptoms.html",{'symptoms': symptoms})

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
import os

def disease_predictor_by_image(request):
    if request.method == "POST":
        action = request.POST.get("action")
        uploaded_file = request.FILES.get("tongue_image")

        if uploaded_file:
            # Save the uploaded image in the 'uploads/' folder inside MEDIA_ROOT
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads/'))
            filename = fs.save(uploaded_file.name, uploaded_file)
                
            # Make sure the file URL includes 'uploads/' directory
            file_url = os.path.join('uploads', filename)  # This ensures the path is relative to 'uploads/'

            if action == "upload":
                    print(os.path.join(settings.MEDIA_ROOT,file_url))
                    diagnosis_result = predict_using_image(os.path.join(settings.MEDIA_ROOT,file_url))
                    return render(request, "tongue.html", {"file_url": file_url,"MEDIA_URL": settings.MEDIA_URL, "message": "Image uploaded successfully !" ,"result": diagnosis_result})

    return render(request, "tongue.html")





def accuracy_checker(request):
    # Evaluate models
    model_metrics = evaluate_models()
    print(model_metrics)
    # Render the data and charts
    return render(request, 'accuracy_checker.html', {'model_metrics': model_metrics})

def extract_doctors(request):

    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client['hospital_database']  # Create/use a database
    collection = db['doctors']  # Create/use a collection

    collection.delete_many({})

    kushal_extract_doctors()

    
   

    # Fetch data from MongoDB
    data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB _id field

    # Convert data to Pandas DataFrame for easier manipulation (if needed)
    df = pd.DataFrame(data)

    # Pass the DataFrame or list to the frontend as context
    context = {
        "doctor_data": data,  # You can pass `data` as a list or `df.to_dict('records')` for DataFrame
    }
    
    return render(request, "doctor_table.html", context)

def disease_precautions(request):
    # Get all diseases to display in the dropdown
    diseases = Disease.objects.all()
    return render(request, "disease_medicine.html", {"diseases": diseases})

def get_medicines(request):
    # Fetch medicines for a specific disease
    disease_id = request.GET.get("disease_id")
    if disease_id:
        medicines = Medicine.objects.filter(disease_id=disease_id).values_list("name", flat=True)
        return JsonResponse({"medicines": list(medicines)})
    return JsonResponse({"medicines": []})