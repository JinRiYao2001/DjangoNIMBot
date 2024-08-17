from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import chat_view, ChatViewSet

router = DefaultRouter()
router.register("chat", ChatViewSet)

urlpatterns = [
    path('chat_ui/', chat_view, name='chat_ui'),
    path('chat_api/', include(router.urls), name='chat_api')
]
