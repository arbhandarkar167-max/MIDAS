from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class Patient(models.Model):

    class Gender(models.TextChoices):
        MALE = "male", "Male"
        FEMALE = "female", "Female"

    class MelanomaHistory(models.TextChoices):
        YES = "yes", "Yes"
        NO = "no", "No"

    age = models.PositiveIntegerField(
        validators=[
            MinValueValidator(0),
            MaxValueValidator(120),
        ]
    )

    gender = models.CharField(
        max_length=20,
        choices=Gender.choices,
    )

    fitzpatrick = models.CharField(
        max_length=200,
    )

    melanoma_history = models.CharField(
        max_length=10,
        choices=MelanomaHistory.choices,
    )

    def __str__(self):
        return f"Patient {self.id}"


class Prediction(models.Model):

    class Gender(models.TextChoices):
        MALE = "male", "Male"
        FEMALE = "female", "Female"

    class MelanomaHistory(models.TextChoices):
        YES = "yes", "Yes"
        NO = "no", "No"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="predictions",
        null=True,
        blank=True,
    )

    patient = models.ForeignKey(
        Patient,
        on_delete=models.CASCADE,
        related_name="predictions",
        null=True,
        blank=True,
    )

    # Patient metadata
    age = models.PositiveIntegerField(
        validators=[
            MinValueValidator(0),
            MaxValueValidator(120),
        ]
    )

    gender = models.CharField(
        max_length=20,
        choices=Gender.choices,
    )

    fitzpatrick = models.CharField(
        max_length=200,
    )

    melanoma_history = models.CharField(
        max_length=10,
        choices=MelanomaHistory.choices,
    )

    race = models.CharField(
        max_length=50,
    )

    # Lesion metadata
    location = models.CharField(
        max_length=50,
    )

    distance = models.CharField(
        max_length=30,
    )

    is_control = models.BooleanField(
        default=False,
    )

    length = models.FloatField(
        validators=[
            MinValueValidator(0),
        ]
    )

    width = models.FloatField(
        validators=[
            MinValueValidator(0),
        ]
    )

    # Original uploaded image
    image = models.ImageField(
        upload_to="lesions/",
    )

    # Grad-CAM visualization
    gradcam_image = models.ImageField(
        upload_to="gradcam/",
        null=True,
        blank=True,
    )

    # ML output
    predicted_class = models.CharField(
        max_length=100,
        blank=True,
        default="",
    )

    confidence = models.FloatField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(0.0),
            MaxValueValidator(1.0),
        ],
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
    )

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Prediction {self.pk} - {self.location}"