import io,os, re
from time import mktime
import parsedatetime as pdt
from datetime import datetime
from datetime import tzinfo, timedelta, datetime
HOUR=timedelta(hours=24)

positions = {"start":0,"end":1}

def processDir(basedir,procF, fileName):
  for nextThing in os.listdir( basedir ):
     path = os.path.join(basedir, nextThing)
     if os.path.isdir(path):
        processDir(path, procF, fileName)
     else:
       if (nextThing == fileName):
          print path
          procF(path)

def getTime(line, inprocessName):
   parts = line.split()
   return (parts[2], datetime.strptime(datetime.now().strftime("%Y/%m/%d ")+ parts[1],"%Y/%m/%d %H:%M:%S.%f"), inprocessName)

def addToSets(sets, whichType, fileName, timeTuple):
    set = [datetime.now(),datetime.now()]
    key = fileName + ":" + timeTuple[0]
    if (sets.has_key(key)):
      set = sets[key]
    set[positions[whichType]] = (timeTuple[1],timeTuple[2])
    sets[key] = set     
      
def processFile(fileName, sets):
   startExp = re.compile("Iteration 0, loss =")
   endExp = re.compile("train_iter_15000.caffemodel")
   fs = open(fileName, "r")
   inprocess = list()
   while True:
     line = fs.readline()
     if (len(line) == 0):
       break
     strippedline = line.strip()
     if (strippedline.startswith("START")):
       inprocess.append(strippedline)
     if (startExp.search(line) != None):
       addToSets(sets,"start",fileName,getTime(line,inprocess[0]))
       inprocess.remove(inprocess[0])
     elif (endExp.search(line) != None):
       addToSets(sets,"end",fileName,getTime(line,"?"))
   fs.close()

def dumpSets(sets):
   for key, value in sets.iteritems():
      delta = value[1][0]-value[0][0]
      if (delta.days < 0 or delta.seconds < 0):
        delta = (value[1][0]+HOUR)-value[0][0]
      ss = key.split("/")
      print  ss[-2] + "," +  value[0][1] + "," + str(delta.seconds)

def main():
   sets=dict()
   processDir(".",lambda x: processFile(x, sets),"run.out")
   dumpSets(sets)

