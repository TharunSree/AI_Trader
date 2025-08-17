# control_panel/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # This makes the dashboard the homepage
    path('', views.dashboard_view, name='dashboard'),
    path('training/', views.training_view, name='training'),
    path('training/job/<int:job_id>/stop/', views.stop_job_view, name='stop_job'),
    path('training/meta/start/', views.start_meta_job_view, name='start_meta_job'),
    path('evaluation/', views.evaluation_view, name='evaluation'),
    path('evaluation/report/<int:job_id>/', views.evaluation_report_view, name='evaluation_report'),
    path('paper-trading/', views.papertrading_view, name='papertrading'),
    path('paper-trading/start/', views.start_trader_view, name='start_trader'),
    path('paper-trading/stop/', views.stop_trader_view, name='stop_trader'),
    path('api/trader-status/', views.trader_status_api, name='trader_status_api'),

    path('real-trading/', views.realtrading_view, name='realtrading'),
    path('real-trading/start/', views.start_real_trader_view, name='start_real_trader'),
    path('real-trading/stop/', views.stop_real_trader_view, name='stop_trader_real'),
    path('settings/', views.settings_view, name='settings'),
    path('api/job-status/', views.job_status_api, name='job_status_api'),
]
