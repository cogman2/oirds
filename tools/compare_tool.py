import numpy as np

class CompareTool:
  totalFP = 0.0
  totalFN=0.0
  totalTP=0.0
  totalTN=0.0
  totalWrongLabel=0.0 

#  def __init__(self):
 
  def getTotals(self):
    precision = 0.0
    recal = 0.0
    f1  = 0.0
    if (self.totalTP < 0.5):
      precision = 0.0
      recall = 0.0
      f1 = 0.0
    else:
      precision=self.totalTP/(self.totalTP+self.totalFP)
      recall=self.totalTP/(self.totalTP+self.totalFN)
      f1=2.0 * (precision*recall / (precision+recall))

    accuracy = (self.totalTP + self.totalTN)/(self.totalTP + self.totalTN + self.totalFP + self.totalFN)
    return (self.totalFP, self.totalFN, self.totalTP,self.totalTN, self.totalWrongLabel, precision, recall,accuracy,f1)

  def compareResults(self,result, gt):
    fp=0.0
    fn=0.0
    tp=0.0
    tn=0.0
    wrongLabel=0.0 
    gt = np.transpose(gt)
    for i in range(0,result.shape[0]):
      for j in range(0,result.shape[1]):
        tn += float(result[i,j] == gt[i,j] and gt[i,j] == 0)
        tp += float(result[i,j] == gt[i,j] and gt[i,j] != 0)
        fp += float(gt[i,j] == 0 and result[i,j] != 0)
        fn += float(gt[i,j] != 0 and result[i,j] == 0)
        wrongLabel += float(result[i,j] != 0 and result[i,j] != gt[i,j] and gt[i,j] != 0)
    if (tp < 0.5):
      precision = 0.0
      recall = 0.0
      f1 = 0.0
    else:
      precision=tp/(tp+fp)
      recall=tp/(tp+fn)
      f1=2.0 * (precision*recall / (precision+recall))
    self.totalFP+=fp
    self.totalTP+=tp
    self.totalTN+=tn
    self.totalFN+=fn
    self.totalWrongLabel+=wrongLabel
    return (fp, fn, tp, tn, wrongLabel, precision, recall,(tp+tn)/(tp+tn+fp+fn),f1)
