from django import forms
from PIL import Image

from .models import Prediction


class PredictionForm(forms.ModelForm):

    FITZPATRICK_CHOICES = [
        (
            "i pale white skin, blue/green eyes, blond/red hair",
            "Type I",
        ),
        (
            "ii fair skin, blue eyes",
            "Type II",
        ),
        (
            "iii darker white skin",
            "Type III",
        ),
        (
            "iv light brown skin",
            "Type IV",
        ),
        (
            "v brown skin",
            "Type V",
        ),
        (
            "vi dark brown or black skin",
            "Type VI",
        ),
    ]

    LOCATION_CHOICES = [
        ("head_neck", "Head / Neck"),
        ("lower_extremity", "Lower Extremity"),
        ("other", "Other"),
        ("trunk", "Trunk"),
        ("upper_extremity", "Upper Extremity"),
    ]

    DISTANCE_CHOICES = [
        ("6in", "6 inches"),
        ("1ft", "1 foot"),
        ("dscope", "Dermatoscope"),
        ("n/a - virtual", "Virtual / Other"),
    ]

    RACE_CHOICES = [
        ("asian", "Asian"),
        (
            "black or african american",
            "Black or African American",
        ),
        ("other", "Other"),
        ("unknown", "Unknown"),
        ("white", "White"),
    ]

    gender = forms.ChoiceField(
        choices=Prediction.Gender.choices,
        label="Sex",
    )

    melanoma_history = forms.ChoiceField(
        choices=Prediction.MelanomaHistory.choices,
        label="Personal History of Melanoma",
    )

    fitzpatrick = forms.ChoiceField(
        choices=FITZPATRICK_CHOICES,
        label="Fitzpatrick Skin Type",
    )

    location = forms.ChoiceField(
        choices=LOCATION_CHOICES,
        label="Lesion Location",
    )

    distance = forms.ChoiceField(
        choices=DISTANCE_CHOICES,
        label="Image Capture Distance",
        initial="n/a - virtual",
    )

    race = forms.ChoiceField(
        choices=RACE_CHOICES,
        label="Race",
        initial="unknown",
    )

    is_control = forms.BooleanField(
        required=False,
        label="Control Image",
    )

    class Meta:
        model = Prediction

        fields = [
            "age",
            "gender",
            "fitzpatrick",
            "melanoma_history",
            "race",
            "location",
            "distance",
            "is_control",
            "length",
            "width",
            "image",
        ]

        widgets = {
            "age": forms.NumberInput(
                attrs={
                    "placeholder": "Enter patient age",
                    "min": 0,
                    "max": 120,
                }
            ),
            "length": forms.NumberInput(
                attrs={
                    "step": "0.1",
                    "min": "0",
                    "placeholder": "Lesion length in mm",
                }
            ),
            "width": forms.NumberInput(
                attrs={
                    "step": "0.1",
                    "min": "0",
                    "placeholder": "Lesion width in mm",
                }
            ),
            "image": forms.ClearableFileInput(
                attrs={
                    "accept": (
                        "image/jpeg,"
                        "image/png,"
                        "image/jpg"
                    ),
                }
            ),
        }

    def clean_image(self):

        image = self.cleaned_data.get(
            "image"
        )

        if not image:
            raise forms.ValidationError(
                "Please upload a lesion image."
            )

        if image.size > 10 * 1024 * 1024:
            raise forms.ValidationError(
                "Image size must be less than 10 MB."
            )

        try:

            img = Image.open(image)
            img.verify()

        except Exception:

            raise forms.ValidationError(
                "Please upload a valid image file."
            )

        # PIL.verify() moves the file pointer.
        # Reset it so the ML predictor can read it.
        image.seek(0)

        return image