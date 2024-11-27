from django import forms
from django import forms

class RegistrationForm(forms.Form):
    username = forms.CharField(max_length=150, required=True)
    email = forms.EmailField(required=True)
    password = forms.CharField(widget=forms.PasswordInput, required=True)
    confirm_password = forms.CharField(widget=forms.PasswordInput, required=True)

class LoginForm(forms.Form):
    username = forms.CharField(max_length=150, required=True)
    password = forms.CharField(widget=forms.PasswordInput, required=True)

class UploadForm(forms.Form):
    purpose_choices = [
        ("bank", "Bank Work"),
        ("government", "Government Work"),
    ]
    purpose = forms.ChoiceField(choices=purpose_choices, required=True)
    image = forms.ImageField(required=True)
