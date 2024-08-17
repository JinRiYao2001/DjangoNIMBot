from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rest_framework.decorators import action
from rest_framework.viewsets import ModelViewSet

from chat.serializers import MessageSerializer
from django.apps import apps

from django_nimbot.settings import LLM_MODEL
from .models import Message
from django.shortcuts import render
from django.http import JsonResponse


def get_bot_response(user_message):
    chat_config = apps.get_app_config('chat')
    CHAIN = (
            {"context": chat_config.retriever, "question": RunnablePassthrough()}
            | chat_config.prompt
            | LLM_MODEL
            | StrOutputParser()
    )
    result = CHAIN.invoke(user_message)
    return result


class ChatViewSet(ModelViewSet):
    serializer_class = MessageSerializer
    queryset = Message.objects.all()

    @action(detail=False, methods=["POST"], permission_classes=[])
    def chat_llm(self, request):
        user_message = request.data.get('message')
        bot_response = get_bot_response(user_message)
        Message.objects.create(user_message=user_message, bot_response=bot_response)
        return JsonResponse({"user_message": user_message, "bot_response": bot_response})


def chat_view(request):
    return render(request, "chat/index.html")
