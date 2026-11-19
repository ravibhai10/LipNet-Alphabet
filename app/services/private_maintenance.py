"""Restart-safe maintenance; durable work and retry state live in SQLite."""
import logging
import threading

from app.settings import settings

_stop = threading.Event()
_thread = None
_lock = threading.Lock()


def start():
    global _thread
    with _lock:
        if _thread and _thread.is_alive():
            return
        from app.adaptive_routes import store, trainer
        from pipeline.adaptive import lifecycle
        trainer.restart_executor()
        lifecycle.recover(store)
        _stop.clear()

        def run():
            while not _stop.is_set():
                try:
                    lifecycle.maintenance(store)
                except Exception:
                    logging.getLogger(__name__).exception('Private maintenance failed; retry scheduled')
                _stop.wait(settings().maintenance_seconds)

        _thread = threading.Thread(target=run, name='private-retention', daemon=True)
        _thread.start()


def stop():
    from app.adaptive_routes import trainer
    trainer.close()
    _stop.set()
    if _thread:
        _thread.join(timeout=5)
