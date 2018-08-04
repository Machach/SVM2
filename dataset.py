import pandas as pd

def arff_to_csv(fpath):
    if fpath.find('.arff')<0:
        print('not a .arff file')
        return
    f=open(fpath)
    lines=f.readlines()
    content=[]
    for l in lines:
        content.append(l)
    datas=[]
    for c in content:
        cs=c.split(',')
        datas.append(cs)


    df=pd.DataFrame(data=datas,index=None,columns=None)
    filename=fpath[:fpath.find('.arff')]+'.csv'
    df.to_csv(filename,index=None)


arff_to_csv('E:\python test\SVM2\datasets\Scene\Scene.arff')


