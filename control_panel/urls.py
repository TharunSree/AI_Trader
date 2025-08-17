# control_panel/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # This makes the dashboard the homepage
    path('', views.dashboard_view, name='dashboard'),
    path('training/', views.training_view, name='training'),
    path('training/job/<int:job_id>/stop/', views.stop_job_view, name='stop_job'),
    path('training/meta/start/', views.start_meta_job_view, name='start_meta_job'),
    # path('paper-trading/', views.papertrading_view, name='papertrading'),
    # path('real-trading/', views.realtrading_view, name='realtrading'),
    # path('settings/', views.settings_view, name='settings'),
]
