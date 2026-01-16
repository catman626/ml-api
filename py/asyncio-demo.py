import asyncio

async def async_foo():
    print("hello")
    await asyncio.sleep(1)
    print("world")

    
async def main():
    await async_foo()

if __name__ == "__main__":
    aobj = main()
    asyncio.run(aobj)