"""
ASGI config for eDETECT project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
"""

# import os

# from django.core.asgi import get_asgi_application

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'eDETECT.settings')

# application = get_asgi_application()


# asgi.py
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from Cards.consumers import FrameConsumer,PeakFrameConsumer
from django.urls import path
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'eDETECT.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": URLRouter(
        [
            # Define WebSocket consumers here
            # For example: re_path(r'ws/some_path/$', SomeConsumer.as_asgi()),
            path('ws/', FrameConsumer.as_asgi(),name="processedFrame"),
            path('ws/Peak', PeakFrameConsumer.as_asgi()),
        ]
    ),
})
