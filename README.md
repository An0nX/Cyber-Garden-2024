# Cyber-Garden-2024
![Kali](https://img.shields.io/badge/Kali-268BEE?style=for-the-badge&logo=kalilinux&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

![GitHub repo size](https://img.shields.io/github/repo-size/An0nX/Cyber-Garden-2024)
![GitHub Tag](https://img.shields.io/github/v/tag/An0nX/Cyber-Garden-2024)

![Пример работы](https://github.com/user-attachments/assets/65d5fe58-a341-4b1b-8b23-0ea46a0e5695)

Этот проект представляет собой Telegram-бота, который принимает PDF-документы, извлекает из них текст и анализирует его на основе предоставленных пользователем инструкций.

## Структура проекта

### [handlers/](handlers)
Директория, содержащая обработчики сообщений и команд.
- **[`start.py`](handlers/start.py)**: Обработчик команд `/start` и `/help`, отправляет приветственное сообщение.
- **[`agent_connector.py`](handlers/agent_connector.py)**: Обработчик для взаимодействия с агентом, настроенным для анализа текста и генерации ответов.

### [utils/](utils)
Утилиты и вспомогательные функции.
- **[`agent.py`](utils/agent.py)**: Класс для работы с OpenAI API и функции для генерации ответов на основе текста.
- **[`pdf_reader.py`](utils/pdf_reader.py)**: Утилиты для работы с PDF-файлами.

### [`aiogram_run.py`](aiogram_run.py)
Основной файл для запуска бота. Содержит функцию `main`, которая включает маршрутизаторы, удаляет существующие вебхуки и запускает опрос бота.

### [`create_bot.py`](create_bot.py)
Файл настройки бота. Содержит инициализацию бота, диспетчера и планировщика.

### [`requirements.txt`](requirements.txt)
Список зависимостей проекта, необходимых для его работы, таких как `aiogram`, `APScheduler`, `numpy`, `openai` и другие.

## Запуск проекта

1. Установите зависимости:
    ```sh
    pip install -r requirements.txt
    ```

2. Запустите бота:
    ```sh
    python aiogram_run.py
    ```

## Пример использования

1. Отправьте команду `/start` или `/help` для получения приветственного сообщения.
2. Загрузите PDF-документ для анализа.
3. Отправьте инструкции для анализа текста.
