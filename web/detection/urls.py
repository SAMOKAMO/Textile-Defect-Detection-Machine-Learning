from django.urls import path
from . import views

urlpatterns = [
    path('',                views.index,       name='index'),
    path('api/predict/',    views.predict_api, name='predict_api'),
    path('api/log/',        views.log_api,     name='log_api'),
    path('api/summary/',    views.summary_api, name='summary_api'),
]
