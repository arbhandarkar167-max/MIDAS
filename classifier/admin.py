from django.contrib import admin
from .models import Prediction, Patient
from django.utils.html import format_html

# Register your models here.

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):

    list_display = [
        "id",
        "age",
        "gender",
        "fitzpatrick",
        "melanoma_history",
    ]

    search_fields = [
        "gender",
    ]

    list_filter = [
        "gender",
        "melanoma_history",
    ]

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "image_preview",
        "age",
        "gender",
        "location",
        "predicted_class",
        "confidence",
        "created_at",
    ]

    search_fields = [
        "location",
        "predicted_class"
    ]

    list_filter = [
        "gender",
        "melanoma_history",
        "created_at",
    ]

    ordering = [
    "-created_at"
    ]

    def image_preview(self, obj):

        if obj.image:
            return format_html(
                '<img src="{}" width="80" />',
                obj.image.url
            )

        return "No image"
   
