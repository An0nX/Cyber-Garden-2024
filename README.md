# PDF Analysis Telegram Bot

Этот проект представляет собой Telegram-бота, который принимает PDF-документы, извлекает из них текст и анализирует его на основе предоставленных пользователем инструкций.

## Структура проекта

### bot/main.py

Основной файл, запускающий Telegram-бота.

- **Импорт библиотек и модулей**:
    ```python
    from aiogram import Bot, Dispatcher, types
    from aiogram.contrib.middlewares.logging import LoggingMiddleware
    from aiogram.utils import executor
    from document_processing.pdf_parser import extract_text_from_pdf
    from text_analysis.analyzer import analyze_text
    from response_generation.formatter import format_response
    import os
    ```

- **Константы и инициализация**:
    ```python
    API_TOKEN = 'YOUR_TELEGRAM_BOT_API_TOKEN'
    bot = Bot(token=API_TOKEN)
    dp = Dispatcher(bot)
    dp.middleware.setup(LoggingMiddleware())
    session_storage = {}
    ```

- **Обработчики сообщений**:
    - **send_welcome**: Обработчик команд `/start` и `/help`, отправляет приветственное сообщение.
        ```python
        @dp.message_handler(commands=['start', 'help'])
        async def send_welcome(message: types.Message):
            await message.reply("Hi! Send me a PDF document with instructions to analyze.")
        ```

    - **handle_document**: Обработчик документов, загружает PDF, извлекает текст и сохраняет его в session_storage.
        ```python
        @dp.message_handler(content_types=[types.ContentType.DOCUMENT])
        async def handle_document(message: types.Message):
            document = message.document
            file_path = await document.download(destination_dir='.')
            extracted_text = extract_text_from_pdf(file_path.name)
            session_storage[message.from_user.id] = extracted_text
            await message.reply("Document received and processed. Send me instructions for analysis.")
        ```

    - **handle_text**: Обработчик текстовых сообщений, анализирует текст из PDF на основе инструкций пользователя.
        ```python
        @dp.message_handler()
        async def handle_text(message: types.Message):
            user_id = message.from_user.id
            if user_id not in session_storage:
                await message.reply("Please upload a PDF document first.")
                return
            instructions = message.text
            extracted_text = session_storage[user_id]
            analysis_result = analyze_text(extracted_text, instructions)
            response = format_response(analysis_result)
            await message.reply(response)
        ```

- **Запуск бота**:
    ```python
    if __name__ == "__main__":
        executor.start_polling(dp, skip_updates=True)
    ```

### document_processing/pdf_parser.py

Модуль для обработки PDF-документов.

- **extract_text_from_pdf**: Функция для извлечения текста из PDF-файла.
    ```python
    def extract_text_from_pdf(file_path):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ''
            for page_num in range(reader.numPages):
                text += reader.getPage(page_num).extract_text()
        return text
    ```

### text_analysis/analyzer.py

Модуль для анализа текста.

- **extract_symptoms**: Функция-заглушка для извлечения симптомов из текста.
    ```python
    def extract_symptoms(text):
        return ["symptom1", "symptom2"]
    ```

- **extract_diagnosis**: Функция-заглушка для извлечения диагноза из текста.
    ```python
    def extract_diagnosis(text):
        return ["diagnosis1", "diagnosis2"]
    ```

- **analyze_text**: Функция для анализа текста на основе инструкций.
    ```python
    def analyze_text(text, instructions):
        if not instructions:
            symptoms = extract_symptoms(text)
            diagnosis = extract_diagnosis(text)
            return {"symptoms": symptoms, "diagnosis": diagnosis}
        else:
            return custom_analysis(text, instructions)
    ```

- **custom_analysis**: Функция-заглушка для кастомного анализа текста.
    ```python
    def custom_analysis(text, instructions):
        return {"custom_analysis": "result"}
    ```

### response_generation/formatter.py

Модуль для форматирования ответа.

- **format_response**: Функция для форматирования анализа в ответный текст.
    ```python
    def format_response(analysis_result):
        if "symptoms" in analysis_result and "diagnosis" in analysis_result:
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(["Symptoms", "Diagnosis"])
    ```
