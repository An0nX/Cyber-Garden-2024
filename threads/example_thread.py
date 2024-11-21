class ExampleTask:
    def __init__(self, name, db):
        self.name = name
        self.db = db

    async def run(self):
        print(f"Task {self.name} is running")
        await self.db.execute("INSERT INTO example_table (name) VALUES ($1)", self.name)
        result = await self.db.fetch("SELECT * FROM example_table")
        print(result)