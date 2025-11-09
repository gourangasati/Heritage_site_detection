from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField()

class GuideBookingForm(forms.Form):
    site_name = forms.CharField(max_length=200, required=True)
    visitor_name = forms.CharField(max_length=200, required=True)
    email = forms.EmailField(required=True)
    phone = forms.CharField(max_length=20, required=True)
    visit_date = forms.DateField(required=True)
    group_size = forms.CharField(max_length=50, required=True)
    language = forms.CharField(max_length=50, required=True)
    special_requirements = forms.CharField(widget=forms.Textarea, required=False)

