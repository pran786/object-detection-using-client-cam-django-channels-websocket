# routing.py
from django.urls import path,re_path
from . import consumers

websocket_urlpatterns = [
    path('ws/', consumers.FrameConsumer.as_asgi(), name='processedFrame'),
    path('ws/Peak', consumers.PeakFrameConsumer.as_asgi(), name='PeakprocessedFrame'),
    # re_path(r'ws/(?P<form_data_id>\d+)/$', consumers.FrameConsumer.as_asgi()),
]
