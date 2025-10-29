from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.index , name="diagnosemeindex"),
    path('about/',views.about,name="about"),
    path('signup/', views.handleSignUp, name="handleSignUp"),
    path('login/', views.handleLogin, name="handleSignUp"),
    path('logout/', views.handleLogout, name="handleSignUp"),
    path('predict-disease/',views.predict,name='predict-disease'),
    path('book-doctor/',views.book_doctor,name="book-doctor"),
    path('disease_predictor_by_symptoms/',views.disease_predictor_by_symptoms,name="disease-predictor-by-symptoms"),
    path('disease-predictor-by-image/',views.disease_predictor_by_image,name="disease-predictor-by-image/"),
    path('accuracy_checker/',views.accuracy_checker,name="accuracy_checker"),
    path('extract_doctors/',views.extract_doctors,name="accuracy_checker"),
    #path('medicine_recommend/',views.medicine_recommend,name="medicine_recommend")
    path("disease-precautions/", views.disease_precautions, name="disease_medicine"),
    path("get-medicines/", views.get_medicines, name="get_medicines")
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
