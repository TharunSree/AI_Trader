import os
import asyncio
import logging
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import trader_project.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'trader_project.settings')

logger = logging.getLogger("trader_project.asgi")
django_asgi_app = get_asgi_application()


async def quiet_http_app(scope, receive, send):
    try:
        await django_asgi_app(scope, receive, send)
    except asyncio.CancelledError:
        logger.debug("HTTP request cancelled by client disconnect.", exc_info=False)
        return
    except RuntimeError as exc:
        if "cannot schedule new futures after shutdown" in str(exc):
            logger.debug("HTTP request ended during executor shutdown.", exc_info=False)
            return
        raise


application = ProtocolTypeRouter({
    "http": quiet_http_app,
    "websocket": AuthMiddlewareStack(
        URLRouter(
            trader_project.routing.websocket_urlpatterns
        )
    ),
})
