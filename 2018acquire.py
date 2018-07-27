import glob,itertools,os
def confirmMaxTrialCount(filePath):
  maxTrailCount = 1
  for fileName in glob.glob(filePath):
    trialCount = 0
    with open(fileName, 'r') as f:
      for line in f.readlines():
        if line.split(']')[0][:9]=='[Mark_End':
          trialCount = trialCount+1  
    if trialCount > maxTrailCount:
      maxTrailCount = trialCount
  return (maxTrailCount+1)   
markDict = {};t=[0,0,0]#;tmin = 400#;;arr=[]
lineCount = [[0,0] for x in range(0, confirmMaxTrialCount('B*.txt'))]
if not os.path.exists('./wingthinks'):
  os.makedirs('./wingthinks')
for fileName in glob.glob('B*.txt'):
  totalLineCount = sum(1 for line in open(fileName))
  lineCount[0][0] = 0;index=0
  count = 0;markCount = 0
  with open(fileName, 'r') as f:
    for line in f.readlines():
      count=count+1
#        try:        
      if len(line.split()) == 1:
        markCount = markCount+1
        index=(markCount+1)//2
        if index == int(line.split(']')[1][2:]):
          if line.split(']')[0][:9] == '[Mark_Sta':
            lineCount[index - 1][0] = count
          elif line.split(']')[0][:9]=='[Mark_End':
            lineCount[index - 1][1] = count
      if count == totalLineCount-1:
        lineCount[index][0]=totalLineCount
  markDict[fileName] = lineCount
  name=fileName.split('.')[0]
  l = len(name);sn = name[l-10:l]
  for sampleIndex in range(index):
    snTest = './wingthinks/'+sn+'_test'+str(sampleIndex)+'.csv'
    snNone = './wingthinks/'+sn+'_none'+str(sampleIndex)+'.csv'
    with open(snTest, 'w') as raw:
      with open(fileName, "r") as fp:
        for line in itertools.islice(fp, markDict[fileName][sampleIndex][0], markDict[fileName][sampleIndex][1]- 1):          
          if len(line.split()) == 4:
            t[0]=float(line.split()[1]) - float(line.split()[0])
            t[1]=float(line.split()[2]) - float(line.split()[0])
            t[2]=float(line.split()[3]) - float(line.split()[0])
          elif len(line.split()) == 6:
            t[0]=float(line.split()[3]) - float(line.split()[2])
            t[1]=float(line.split()[4]) - float(line.split()[2])
            t[2]=float(line.split()[5]) - float(line.split()[2])
          raw.writelines([str(t[0])+','+str(t[1])+','+str(t[2])+'\r\n'])
    with open(snNone, 'w') as raw:
      with open(fileName, "r") as fp:
        for line in itertools.islice(fp, markDict[fileName][sampleIndex][1], markDict[fileName][sampleIndex+1][0] - 1):          
          if len(line.split()) == 4:
            t[0]=float(line.split()[1]) - float(line.split()[0])
            t[1]=float(line.split()[2]) - float(line.split()[0])
            t[2]=float(line.split()[3]) - float(line.split()[0])
          elif len(line.split()) == 6:
            t[0]=float(line.split()[3]) - float(line.split()[2])
            t[1]=float(line.split()[4]) - float(line.split()[2])
            t[2]=float(line.split()[5]) - float(line.split()[2])
          raw.writelines([str(t[0])+','+str(t[1])+','+str(t[2])+'\r\n'])  
  snNone = './wingthinks/'+sn+'_none'+str(index)+'.csv'
  with open(snNone, 'w') as raw:
    with open(fileName, "r") as fp:
      for line in itertools.islice(fp, 0, markDict[fileName][0][0] - 1):          
        if len(line.split()) == 4:
          t[0]=float(line.split()[1]) - float(line.split()[0])
          t[1]=float(line.split()[2]) - float(line.split()[0])
          t[2]=float(line.split()[3]) - float(line.split()[0])
        elif len(line.split()) == 6:
          t[0]=float(line.split()[3]) - float(line.split()[2])
          t[1]=float(line.split()[4]) - float(line.split()[2])
          t[2]=float(line.split()[5]) - float(line.split()[2])
        raw.writelines([str(t[0])+','+str(t[1])+','+str(t[2])+'\r\n']) 
for emptyFile in glob.glob('./wingthinks/*.csv'):
  if os.stat(emptyFile).st_size == 0:
    os.remove(emptyFile)
                #linecache.clearcache()
#print(markDict)
#def biggerValue(a,b):
#  if a > b:
#    return a
#  return b
#def readLineAndWriteToFile(sourceFileName, targetFileName, start, end):
#  #linecache.checkcache(sourceFileName)
#  with open(targetFileName, 'w') as f:
#    f.write(linecache.getline(sourceFileName)[start : end-1])
#  for i in range(22):
#    allZero = allZero+lineCount[i][0]+lineCount[i][1]
#  if allZero == 0:
#    continue
          #if line[:6] != '[Mark_':
          #print('mark:%s %d %s' % (fileName, count, line))          
        #raw.writelines([str(t[0])+','+str(t[1])+','+str(t[2])+'\r\n'])
      #if markCount > maxMarkCount:
        #maxMarkCount = markCount
        #print('markCount {}'.format(markCount))
        #markCount=0
      #print('max mark count:%d' % maxMarkCount)
#        except Exception as e:
#          print('%s %d %s' %(fileName,count,line))
           
'''if sn not in arr:
    arr.append(sn)
  else:
    print(sn)
print('*'*80);print(arr)'''

#count = count+1
#print('%d %s' % (count,file))
