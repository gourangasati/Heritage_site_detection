# mian/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict_view, name='predict'),
    path('search/', views.search, name='search'),
    path('site/<str:site_key>/', views.site_detail, name='site_detail'),
    path('guide/', views.guide, name='guide'),
    path('about/', views.about, name='about'),
    path('profile/', views.profile, name='profile'),
]
