import sys
import time


for i in range(10):
    sys.stdout.write('\r' + 'Прогресс: %d' % i)
    sys.stdout.flush()
    time.sleep(0.5)
