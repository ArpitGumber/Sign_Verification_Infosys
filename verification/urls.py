# verification/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Login page
    path('registration/', views.registration, name='registration'),  # Registration page
    path('upload/', views.upload_view, name='upload'),  # Upload page (update here)
    path('login/', views.login_view, name='login'),  # Login page
    path('logout/', views.logout_view, name='logout'),  # Logout path
    path('predict/', views.predict_signature, name='predict'),
]