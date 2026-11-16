"""Bound CPU request concurrency across both independent word models."""
import threading
word_cpu_lock = threading.Lock()
