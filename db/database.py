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
        