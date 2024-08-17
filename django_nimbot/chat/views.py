import base64

import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rest_framework.decorators import action
from rest_framework.viewsets import ModelViewSet

from chat.serializers import MessageSerializer
from django.apps import apps

from django_nimbot.settings import LLM_MODEL, NVIDIA_API_KEY, LLM_MODEL_RAG
from .models import Message
from django.shortcuts import render
from django.http import JsonResponse


def get_bot_response(user_message):
    chat_config = apps.get_app_config('chat')
    # CHAIN = (
    #         {"context": chat_config.retriever, "question": RunnablePassthrough()}
    #         | chat_config.prompt
    #         | LLM_MODEL
    #         | StrOutputParser()
    # )
    result = LLM_MODEL.invoke(user_message)
    return result.content


def get_bot_response_rag(user_message):
    chat_config = apps.get_app_config('chat')
    CHAIN = (
            {"context": chat_config.retriever, "question": RunnablePassthrough()}
            | chat_config.prompt
            | LLM_MODEL_RAG
            | StrOutputParser()
    )
    result = CHAIN.invoke(user_message)
    return result


def get_image_from_prompt(prompt):
    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/sdxl-turbo"

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
    }

    payload = {
        "text_prompts": [{
            "text": prompt}],
        "seed": 0,
        "sampler": "K_EULER_ANCESTRAL",
        "steps": 2
    }
    response = requests.post(invoke_url, headers=headers, json=payload)
    response.raise_for_status()
    response_body = response.json()
    return response_body.get('artifacts')[0].get('base64')


class ChatViewSet(ModelViewSet):
    serializer_class = MessageSerializer
    queryset = Message.objects.all()

    @action(detail=False, methods=["POST"], permission_classes=[])
    def chat_llm(self, request):
        user_message = request.data.get('message')
        bot_response = get_bot_response(user_message)
        Message.objects.create(user_message=user_message, bot_response=bot_response)
        return JsonResponse({"user_message": user_message, "bot_response": bot_response})

    @action(detail=False, methods=["POST"], permission_classes=[])
    def chat_llm_rag(self, request):
        user_message = request.data.get('message')
        bot_response = get_bot_response_rag(user_message)
        Message.objects.create(user_message=user_message, bot_response=bot_response)
        return JsonResponse({"user_message": user_message, "bot_response": bot_response})

    @action(detail=False, methods=["POST"], permission_classes=[])
    def generate_image(self, request):
        prompt = request.data.get('prompt')
        image_binary = get_image_from_prompt(prompt)
        Message.objects.create(user_message=prompt, bot_response=image_binary)
        return JsonResponse({"image_base64": image_binary})


def chat_view(request):
    return render(request, "chat/index.html")
