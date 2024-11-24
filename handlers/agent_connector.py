from aiogram import Router
from aiogram.types import Message, BufferedInputFile
import io
from asyncio import to_thread
from decouple import config
from create_bot import bot

from utils.agent import LLMAgent

file_router = Router()

@file_router.message(lambda message: message.document and message.document.mime_type == 'application/pdf')
async def handle_pdf_file(message: Message):
    document = message.document
    file_bytes = io.BytesIO()
    await bot.download(document, destination=file_bytes)
    text_caption = message.caption or None

    llm_agent = LLMAgent(api_key=config("OPENAI_API_KEY"))

    response = await llm_agent.process(text_input=text_caption, documents=file_bytes)

    if response.get("error"):
        await message.reply(f"Errors occurred during processing: {response["error"]}")
    else:
        if len(response["text"]) > 4096:
            buffer = io.BytesIO()
            buffer.write(response["text"].encode())
            buffer.seek(0)
            text_file = BufferedInputFile(buffer, filename="response.txt")
            await message.answer_document(text_file)
        else:
            await message.reply(response["text"])

        if response["csv_buffer"]:
            response["csv_buffer"].seek(0)  # Сбрасываем указатель на начало файла
            csv_file = BufferedInputFile(response["csv_buffer"], filename="result.csv")
            await message.answer_document(csv_file)


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
