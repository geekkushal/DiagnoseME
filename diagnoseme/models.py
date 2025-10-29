from django.db import models

# Create your models here.
from django.contrib.auth.models import User

class PatientAppointment(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="appointment")
    booked_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
           return f"Appointment for {self.user.username} ,booked on {self.booked_at}"

class Disease(models.Model):
    name = models.CharField(max_length=100)
    def __str__(self):
        return f"{self.name}"

class Medicine(models.Model):
    name = models.CharField(max_length=100)
    disease = models.ForeignKey(Disease, on_delete=models.CASCADE, related_name='medicines')

    def __str__(self):
        return f"For disease - {self.disease} , medicine - {self.name}"