from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
)


def get_main_keyboard():
    """
    Return a ReplyKeyboardMarkup object with two buttons: 
    "Загрузить документ" and "Анализировать текст".
    """
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = [
        KeyboardButton(text="Загрузить документ"),
        KeyboardButton(text="Анализировать текст"),
    ]
    keyboard.add(*buttons)
    return keyboard
