from django.contrib import admin

# Register your models here.
from .models import PatientAppointment
from .models import Disease
from .models import Medicine

admin.site.register(PatientAppointment)
admin.site.register(Disease)
admin.site.register(Medicine)