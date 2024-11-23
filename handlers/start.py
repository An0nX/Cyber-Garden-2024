from aiogram import Router
from aiogram.filters import CommandStart, Command
from aiogram.types import Message

from keyboards.start import get_main_keyboard

start_router = Router()


@start_router.message(CommandStart())
async def cmd_start(message: Message):
    """
    Handles the /start command and sends a welcome message to the user.

    This function is triggered when a user sends the /start command to the bot.
    It responds with a greeting message and prompts the user to send a PDF
    document for analysis. A custom keyboard is also presented to the user.

    Args:
        message (Message): The incoming message object containing user data.
    """
    await message.answer(
        "Здравствуйте! Я бот для анализа PDF-документов. Пожалуйста, отправьте мне PDF-файл с инструкциями.",
    )
