from django.urls import path
from . import views

# Create your URL configuration here.
urlpatterns = [
    path("", views.enter, name = "enter"),
    path("describe/", views.describe, name = "describe"),
    path('predict/', views.predict, name = "predict"),
]