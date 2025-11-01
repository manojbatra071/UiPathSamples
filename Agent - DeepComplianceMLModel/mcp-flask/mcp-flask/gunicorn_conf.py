import multiprocessing

bind = "0.0.0.0:8080"
workers = int((multiprocessing.cpu_count() * 2) + 1)
worker_class = "gthread"
threads = 4
timeout = 120
accesslog = '-'
errorlog = '-'
