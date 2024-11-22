import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.utils import executor
from document_processing.pdf_parser import extract_text_from_pdf
from text_analysis.analyzer import analyze_text
from response_generation.formatter import format_response
import os

API_TOKEN = 'YOUR_TELEGRAM_BOT_API_TOKEN'

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

session_storage = {}

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply("Hi! Send me a PDF document with instructions to analyze.")

@dp.message_handler(content_types=[types.ContentType.DOCUMENT])
async def handle_document(message: types.Message):
    document = message.document
    file_path = await document.download(destination_dir='.')

    # Extract text from PDF
    extracted_text = extract_text_from_pdf(file_path.name)

    # Store the extracted text in session storage
    session_storage[message.from_user.id] = extracted_text

    await message.reply("Document received and processed. Send me instructions for analysis.")

@dp.message_handler()
async def handle_text(message: types.Message):
    user_id = message.from_user.id
    if user_id not in session_storage:
        await message.reply("Please upload a PDF document first.")
        return

    instructions = message.text
    extracted_text = session_storage[user_id]

    # Analyze text based on instructions
    analysis_result = analyze_text(extracted_text, instructions)

    # Format response
    response = format_response(analysis_result)

    await message.reply(response)

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
    