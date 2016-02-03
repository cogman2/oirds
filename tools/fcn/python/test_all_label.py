import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import net_tool
import os
import test_label
import thread
import time
import multiprocessing
from multiprocessing import Process
import threading
import Queue
import json_tools


q = Queue.Queue()
result = dict()

def main():
    from pathlib import Path
    import os
    from multiprocessing import Process
    import sys
    if sys.version_info[0] < 3:
        from StringIO import StringIO
    else:
        from io import StringIO

    for dir in os.listdir( '.' ):
       if (dir.startswith( 'conv' )):
          q.put(dir)

    if len(sys.argv) < 2:
      print "Usage: ", sys.argv[0], " config "
      sys.exit( 0 )

    config = json_tools.loadConfig(sys.argv[1])
    cwd = os.getcwd()

    # Create two threads as follows
    # Create new threads
    thread1 = myThread(1, "Thread-1", sys.argv[1])
    thread2 = myThread(2, "Thread-2", sys.argv[1])

    # Start new Threads
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()

    statsFileName = json_tools.getStatsFileName(config)

    for dir in os.listdir( '.' ):
       if (dir.startswith( 'conv' )):
         print getStats(os.path.join(cwd,dir) + '/' + statsFileName,dir)

class myThread (threading.Thread):
    configFileName=''
    statsFileName=''
    cwd=''
    def __init__(self, threadID, name, configFileName):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.configFileName = configFileName
        self.cwd = os.getcwd()

    def run(self):
      while (not q.empty()):
        try:
          dir = q.get(5)       
          p = Process(target=testLabel, args=(os.path.join(self.cwd,dir),self.configFileName,))
          p.start()
          p.join()        
        except Queue.Empty:
          break

def testLabel(dir, configFileName):
   os.chdir(dir)
   print dir
   test_label.runTest("../" + configFileName) 

def getStats(statsFileName, runName):
     if (os.path.isfile(statsFileName)):
       myvarfile = open(statsFileName, 'r')
       lst = myvarfile.readlines()
       myvarfile.close
       return lst[-1].replace('Totals',runName)
     return runName+',0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0'
     

if __name__=="__main__":
    main()
