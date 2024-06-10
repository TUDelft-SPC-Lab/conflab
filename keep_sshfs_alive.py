from constants import RAW_VIDEOS_FOLDER_IN_STAFF_BULK
import time
from datetime import datetime
 
ping_period_in_seconds: float = 10.0
 
while True:
    print(f"===== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}====")
    file_path = list(RAW_VIDEOS_FOLDER_IN_STAFF_BULK.glob('*'))[0]
    print(file_path)
    time.sleep(ping_period_in_seconds)