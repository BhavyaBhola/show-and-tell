from .views import UploadView
from django.urls import path

urlpatterns = [
    path('gen/', UploadView.as_view()),
]
