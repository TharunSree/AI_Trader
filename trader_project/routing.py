from django.urls import re_path
from control_panel.consumers import DashboardStreamConsumer

websocket_urlpatterns = [
    re_path(r'ws/dashboard/$', DashboardStreamConsumer.as_asgi()),
]