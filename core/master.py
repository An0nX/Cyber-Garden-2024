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
    