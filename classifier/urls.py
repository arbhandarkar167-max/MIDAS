from django.urls import path

from . import views


app_name = "classifier"


urlpatterns = [
    path("", views.home, name="home"),

    path(
        "predict/",
        views.predict_view,
        name="predict",
    ),

    path(
        "about/",
        views.about,
        name="about",
    ),

    path(
        "history/",
        views.history,
        name="history",
    ),

    path(
        "prediction/<int:pk>/",
        views.prediction_detail,
        name="prediction_detail",
    ),

    path(
        "login/",
        views.login_view,
        name="login",
    ),

    path(
        "logout/",
        views.logout_view,
        name="logout",
    ),
]