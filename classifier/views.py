from io import BytesIO

from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.core.files.base import ContentFile
from django.core.paginator import Paginator
from django.shortcuts import (
    get_object_or_404,
    redirect,
    render,
)

from .forms import PredictionForm
from .ml.predictor import predict
from .models import Prediction


def home(request):

    return render(
        request,
        "classifier/home.html",
        {
            "project_name": "MIDAS",
            "description": (
                "Multimodal skin lesion classification "
                "using lesion images and clinical metadata."
            ),
        },
    )


def about(request):

    return render(
        request,
        "classifier/about.html",
        {
            "title": "About MIDAS",
            "description": (
                "MIDAS is a multimodal skin lesion "
                "classification application that combines "
                "lesion imagery with clinical metadata."
            ),
        },
    )


@login_required
def predict_view(request):

    if request.method == "POST":

        form = PredictionForm(
            request.POST,
            request.FILES,
        )

        if form.is_valid():

            prediction = form.save(
                commit=False
            )

            prediction.user = request.user

            try:

                uploaded_image = (
                    form.cleaned_data["image"]
                )

                result = predict(
                    uploaded_image,
                    form.cleaned_data,
                )

                prediction.predicted_class = (
                    result["predicted_class"]
                )

                prediction.confidence = (
                    result["confidence"]
                )

                # Save the prediction first.
                prediction.save()

                # -------------------------------------------------
                # Save Grad-CAM overlay
                # -------------------------------------------------

                gradcam_overlay = (
                    result["gradcam_overlay"]
                )

                buffer = BytesIO()

                gradcam_overlay.save(
                    buffer,
                    format="JPEG",
                    quality=90,
                )

                gradcam_filename = (
                    f"prediction_{prediction.pk}_gradcam.jpg"
                )

                prediction.gradcam_image.save(
                    gradcam_filename,
                    ContentFile(
                        buffer.getvalue()
                    ),
                    save=True,
                )

                messages.success(
                    request,
                    "Lesion analysis completed successfully.",
                )

                return redirect(
                    "classifier:prediction_detail",
                    pk=prediction.pk,
                )

            except FileNotFoundError as error:

                messages.error(
                    request,
                    str(error),
                )

            except Exception as error:

                messages.error(
                    request,
                    f"Prediction failed: {error}",
                )

    else:

        form = PredictionForm()

    return render(
        request,
        "classifier/predict.html",
        {
            "form": form,
        },
    )


@login_required
def history(request):

    queryset = Prediction.objects.filter(
        user=request.user
    ).order_by("-created_at")

    paginator = Paginator(
        queryset,
        10,
    )

    page_number = request.GET.get(
        "page"
    )

    page_obj = paginator.get_page(
        page_number
    )

    return render(
        request,
        "classifier/history.html",
        {
            "predictions": page_obj.object_list,
            "page_obj": page_obj,
        },
    )


@login_required
def prediction_detail(
    request,
    pk,
):

    prediction = get_object_or_404(
        Prediction,
        pk=pk,
        user=request.user,
    )

    return render(
        request,
        "classifier/prediction_detail.html",
        {
            "prediction": prediction,
        },
    )


def login_view(request):

    if request.user.is_authenticated:

        return redirect(
            "classifier:home"
        )

    if request.method == "POST":

        form = AuthenticationForm(
            request,
            data=request.POST,
        )

        if form.is_valid():

            login(
                request,
                form.get_user(),
            )

            next_url = request.POST.get(
                "next"
            )

            if next_url:
                return redirect(next_url)

            next_url = request.GET.get(
                "next"
            )

            if next_url:
                return redirect(next_url)

            return redirect(
                "classifier:home"
            )

    else:

        form = AuthenticationForm()

    return render(
        request,
        "classifier/login.html",
        {
            "form": form,
        },
    )


def logout_view(request):

    logout(request)

    return redirect(
        "classifier:home"
    )