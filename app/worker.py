"""The rq worker that picks up ingested batches.

Started with `python -m app.worker`, or by the worker service in
docker-compose.yml.
"""

from redis import Redis
from rq import Queue, Worker

from .config import settings

QUEUES = ["default"]


if __name__ == "__main__":
    redis_conn = Redis.from_url(settings.redis_url)
    # The connection is passed straight to the Worker rather than pushed with
    # `with Connection(redis_conn)`. Connection is deprecated in rq 1.x and gone
    # in 2.0, so the old form would have broken the worker on the first
    # unpinned upgrade, in a process nothing in the test suite starts.
    queues = [Queue(name, connection=redis_conn) for name in QUEUES]
    Worker(queues, connection=redis_conn).work()
