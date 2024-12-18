import asyncio
from create_bot import bot, dp, scheduler
from handlers.start import start_router
from handlers.agent_connector import file_router

from work_time.time_func import collect_garbage


async def main():
    """
    Main function to start the bot.

    This function sets up the bot by including the start_router,
    deleting any existing webhook and starting the polling.
    """

    scheduler.add_job(collect_garbage, "interval", seconds=10)
    scheduler.start()
    dp.include_router(start_router)
    dp.include_router(file_router)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
