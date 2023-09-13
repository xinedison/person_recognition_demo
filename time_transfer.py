import time
#timestamp = 1686251700
#time_local = time.localtime(timestamp)
#dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
#print(dt)


dt = "2023-04-12 00:00:00"
timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
timestamp = time.mktime(timeArray)
print(timestamp)
