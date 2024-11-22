from aiogram import Router
from aiogram.types import Message
import io

file_router = Router()

@file_router.message(lambda message: message.document.mime_type == 'application/pdf')
async def handle_pdf_file(message: Message):
    document = await message.document.get_file()
    file_bytes = await document.download(destination=io.BytesIO())
    text_caption = message.caption or None
    

    # TODO: extract PDF file data using function in utils/document_processing/pdf_parser.py

    await message.answer(f"PDF extracted successfully")

