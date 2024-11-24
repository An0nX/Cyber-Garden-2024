from aiogram import Router
from aiogram.types import Message, BufferedInputFile
import io
from asyncio import to_thread
from decouple import config
from create_bot import bot
import os
from utils.agent import LLMAgent

file_router = Router()


@file_router.message(
    lambda message: message.document and message.document.mime_type == "application/pdf"
)
async def handle_pdf_file(message: Message):
    document = message.document
    file_bytes = io.BytesIO()
    await bot.download(document, destination=file_bytes)
    text_caption = message.caption or None

    llm_agent = LLMAgent(api_key=config("OPENAI_API_KEY"))

    response = await llm_agent.process(text_input=text_caption, documents=file_bytes)

    if response.get("error"):
        await message.reply(f"Errors occurred during processing: {response['error']}")
    else:
        if len(response["text"]) > 4096:
            # Создаем буфер для текстового файла
            buffer = io.BytesIO()
            buffer.write(response["text"].encode("utf-8"))  # Кодируем текст в байты
            buffer.seek(0)  # Сбрасываем указатель в начало
            text_file = BufferedInputFile(buffer, filename="response.txt")
            await message.answer_document(text_file)  # Отправляем файл
        else:
            await message.reply(response["text"])  # Отправляем текст

        # Отправка CSV-файла
        if response["csv_buffer"]:
            # Сохраняем данные CSV в файл на устройстве
            await message.reply(response["csv_buffer"])


@file_router.message()
async def handle_text(message: Message):
    text = message.text
    llm_agent = LLMAgent(api_key=config("OPENAI_API_KEY"))
    response = await llm_agent.process(text_input=text)

    if response.get("error"):
        await message.reply(f"Errors occurred during processing: {response['error']}")
    else:
        if len(response["text"]) > 4096:
            buffer = io.BytesIO()
            buffer.write(response["text"].encode())
            buffer.seek(0)
            text_file = BufferedInputFile(buffer, filename="response.txt")
            await message.answer_document(text_file)
        else:
            await message.reply(response["text"])
