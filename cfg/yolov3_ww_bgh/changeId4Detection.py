import os
import sys

fileDir = '/home/ubuntu/CV/data/wework_activity/fake_detection/fake_train/'
outDir = '/home/ubuntu/CV/data/wework_activity/fake_detection/fake_train_txt/'


files = os.listdir(fileDir)
total_sz = len(files)

pcnt = 0
for filename in files:
    portion = os.path.splitext(filename)
    pcnt += 1
    # 如果后缀是.txt
    if portion[1] == ".txt":
        filePath = fileDir + filename
        outfilePath = outDir + filename
        with open(filePath,'r') as fp:
            outfp = open(outfilePath,'w+')
            for line in fp:
                strSplit = ' '
                lstline = line.split(strSplit)
                if lstline[0] != '\n':
                    lstline[0] = '0'
                    nline = strSplit.join(lstline)
                    outfp.write(nline)
            outfp.close()
        fp.close()
        
    if pcnt%100 == 0:
        print( str(pcnt/total_sz*100) + '% processed...\n', )