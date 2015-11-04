#!/usr/bin python
# from 'ls -t /tmp | grep $USER.log.INFO.201510 | xargs cat'
import os
import pandas as pd

logs = []
accuracy = []
for fname in os.listdir('/tmp'):
    # Select the October log files.
    if 'hess.log.INFO.201511' in fname:
        f = open('/tmp/'+fname, 'r').read().split('\n')
        for i, line in enumerate(f):
            if 'accuracy =' in line:
                accuracy += [line]
                logs += f[i:i+5]
                    
print('Logs\n')
for log in logs:
    print(log)

print('\n\nAccuracy\n')
for acc in accuracy:
    print(acc)



