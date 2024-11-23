# PDF Analysis Telegram Bot

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

### [keyboards/](keyboards)
Модуль для создания клавиатур в Telegram.
- **[`start.py`](keyboards/start.py)**: Функции для создания и настройки клавиатур.

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