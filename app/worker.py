import os
from redis import Redis
from rq import Worker, Queue, Connection

from .config import settings


listen = ["default"]


if __name__ == "__main__":
    redis_conn = Redis.from_url(settings.redis_url)
    with Connection(redis_conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()
