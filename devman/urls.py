from django.urls import path

from . import views

app_name = 'devman'
urlpatterns = [
    path('', views.overview, name='overview'),
    path('<int:device_id>/', views.detail, name='detail'),
    path('<int:device_id>/repair/', views.repair, name='repair'),
    path('<int:device_id>/error/', views.error, name='error'),
    path('plan/', views.plan, name='plan'),
    path('hist/', views.hist, name='hist'),
    path('deverr/', views.device_error, name='deverr'),
    path('devrep/', views.device_repairing, name='devrep'),
    path('devrun/', views.device_running, name='devrun'),
    path('devstp/', views.device_stop, name='devstp'),
]
