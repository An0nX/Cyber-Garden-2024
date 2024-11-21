# Cyber-Garden-2024

## Структура проекта

Проект состоит из следующих файлов и директорий:

### 1. `core/master.py`

Этот файл является точкой входа в приложение. Он отвечает за инициализацию и запуск основных задач.

#### Содержимое:
- Импорт необходимых модулей и библиотек.
- Функция `main()`, которая создает экземпляры задач, подключается к базе данных и запускает задачи параллельно.
- Блок `if __name__ == "__main__"`, который запускает `main()` асинхронно.

#### Пример кода:
```python
import asyncio
import sys
import os

for root, dirs, files in os.walk(os.path.dirname(__file__) + '/..'):
    for dir in dirs:
        sys.path.append(os.path.abspath(os.path.join(root, dir)))

from db.database import Database
from threads.example_thread import ExampleTask
from core.logger import LogsManager

async def main():
    db = Database(user='youruser', password='yourpassword', database='yourdatabase')

    await db.connect()

    task1 = ExampleTask(name="Task1", db=db)
    task2 = ExampleTask(name="Task2", db=db)

    await asyncio.gather(task1.run(), task2.run())

    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. `db/database.py`

Этот файл содержит класс `Database`, который управляет подключением и взаимодействием с базой данных PostgreSQL.

#### Содержимое:
- Класс `Database` с методами для подключения, закрытия, выполнения запросов и получения данных из базы данных.

#### Пример кода:
```python
import asyncpg

class Database:
    def __init__(self, user, password, database, host='localhost', port=5432):
        self.user = user
        self.password = password
        self.database = database
        self.host = host
        self.port = port
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            user=self.user,
            password=self.password,
            database=self.database,
            host=self.host,
            port=self.port
        )

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def execute(self, query, *args):
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                result = await connection.execute(query, *args)
                return result

    async def fetch(self, query, *args):
        async with self.pool.acquire() as connection:
            result = await connection.fetch(query, *args)
            return result

    async def fetchrow(self, query, *args):
        async with self.pool.acquire() as connection:
            result = await connection.fetchrow(query, *args)
            return result
```

### 3. `threads/example_thread.py`

Этот файл содержит класс `ExampleTask`, который представляет собой пример задачи, выполняемой параллельно.

#### Содержимое:
- Класс `ExampleTask` с методами инициализации и выполнения задачи.

#### Пример кода:
```python
class ExampleTask:
    def __init__(self, name, db):
        self.name = name
        self.db = db

    async def run(self):
        print(f"Task {self.name} is running")
        await self.db.execute("INSERT INTO example_table (name) VALUES ($1)", self.name)
        result = await self.db.fetch("SELECT * FROM example_table")
        print(result)
```

### 4. `core/logger.py`

Этот файл отвечает за настройку и управление логированием в приложении.

#### Содержимое:
- Класс `LogsManager` с методами для подключения лог-файла и получения экземпляра логгера.

#### Пример кода:
```python
class LogsManager:
    def __init__(self):
        import loguru
        self._logger = loguru.logger

    def connect_log_file(self):
        self._logger.add("../logs/{time}.log", rotation="00:00", encoding="utf-8", enqueue=True, level="DEBUG")

    @property
    def logger(self):
        return self._logger
```

### 5. `utils/helpers.py`

Этот файл содержит вспомогательные функции и классы для работы с логами и другими утилитами.

#### Содержимое:
- Класс `LogKeeper`, который отвечает за чтение и фильтрацию лог-файлов.
- Асинхронные функции для получения списка лог-файлов и чтения логов.

#### Пример кода:
```python
import os
import dateutil.parser
import loguru

class LogKeeper:
    def __init__(self, pattern: str, caster_dict: dict):
        self.pattern = pattern
        self.caster_dict = caster_dict

    async def get_log_files(self) -> list[str]:
        """
        Асинхронно собирает список всех лог-файлов в указанной директории.
        """
        log_files_names = []
        for root, dirs, files in os.walk("../logs"):
            for file in files:
                if file.endswith(".log"):
                    log_files_names.append(os.path.join(root, file))
        return log_files_names

    async def read_log_file(
        self,
        file_path: str,
        pattern: str | None = None,
        caster_dict: dict | None = None,
        offset: int = 0,
        from_time: int = 0,
        log_level: int | None = None,
    ) -> str:
        """
        Асинхронно читает лог-файл, фильтруя записи по времени и уровню логирования.
        """
        logs = []
        with open(file_path, "r") as file:
            file.seek(offset)  # Устанавливаем смещение
            for groups in loguru.logger.parse(file, pattern, cast=caster_dict):
                try:
                    # Фильтрация по времени и уровню
                    if groups["time"].timestamp() >= from_time and (
                        log_level is None or groups["level"] == log_level
                    ):
                        logs.append(groups["message"])
                except Exception as e:
                    loguru.logger.warning(f"Ошибка обработки записи: {e}")
        return "\n".join(logs)

# Пример использования
if __name__ == "__main__":
    pattern = r"(?P<time>.*) - (?P<level>[0-9]+) - (?P<message>.*)"  # Паттерн регулярного выражения
    caster_dict = {
        "time": dateutil.parser.parse,  # Каст времени в datetime
        "level": int,  # Каст уровня в int
    }

    log_keeper = LogKeeper(pattern, caster_dict)

    # Пример асинхронного вызова
    import asyncio

    async def main():
        log_files = await log_keeper.get_log_files()
        for log_file in log_files:
            parsed_logs = await log_keeper.read_log_file(log_file, from_time=1672531200, log_level=30)
            print(parsed_logs)
    
    asyncio.run(main())
```

### 6. `requirements.txt`

Этот файл содержит список всех зависимостей, необходимых для работы проекта. Он используется для установки всех необходимых пакетов с помощью pip.

#### Содержимое:
```
asyncpg
loguru
python_dateutil
dotenv
```

### Установка и запуск

Для установки зависимостей и запуска проекта выполните следующие шаги:

1. Склонируйте репозиторий:
    ```sh
    git clone https://example.com/cyber-garden-2024.git
    cd cyber-garden-2024
    ```

2. Создайте и активируйте виртуальное окружение:
    ```sh
    python -m venv venv
    source venv/bin/activate  # Для Windows используйте `venv\Scripts\activate`
    ```

3. Установите зависимости:
    ```sh
    pip install -r requirements.txt
    ```

4. Запустите приложение:
    ```sh
    python core/master.py
    ```

## Заключение

Это README предоставляет обзор структуры проекта и назначения каждого файла. Используйте предоставленные примеры кода и инструкции для настройки и запуска проекта. Если у вас возникнут вопросы или проблемы, пожалуйста, создайте issue в репозитории или обратитесь к документации используемых библиотек.
