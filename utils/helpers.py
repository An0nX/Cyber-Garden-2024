import os
from loguru import logger
import dateutil.parser


class LogKeeper:
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
            for groups in logger.parse(file, pattern, cast=caster_dict):
                try:
                    # Фильтрация по времени и уровню
                    if groups["time"].timestamp() >= from_time and (
                        log_level is None or groups["level"] == log_level
                    ):
                        logs.append(groups["message"])
                except Exception as e:
                    logger.warning(f"Ошибка обработки записи: {e}")
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
            parsed_logs = await log_keeper.read_log_file(
                log_file, from_time=1672531200, log_level=30
            )
            print(parsed_logs)

    asyncio.run(main())
