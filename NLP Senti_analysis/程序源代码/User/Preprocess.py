#文本预处理 完成文本分词、清洗、停用词删除与词向量构建
import jieba
jieba.set_dictionary("./dict.txt")
jieba.initialize()
#import jieba.analyse
import codecs,sys,string,re
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import logging
import os.path
import numpy as np
import pandas as pd
import gensim

# 文本分词
def prepareData(sourceFile,targetFile):
    f = codecs.open(sourceFile, 'r', encoding='utf-8')
    target = codecs.open(targetFile, 'w', encoding='utf-8')
    print ('open source file: '+ sourceFile)
    print ('open target file: '+ targetFile)

    lineNum = 1
    line = f.readline()
    while line:
        print ('---processing ',lineNum,' article---')
        line = clearTxt(line)
        seg_line = sent2word(line)
        target.writelines(seg_line + '\n')       
        lineNum = lineNum + 1
        line = f.readline()
    print ('well done.')
    f.close()
    target.close()

# 清洗文本
def clearTxt(line):
    if line != '': 
        line = line.strip()
        #去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]","",line)
        #去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+", "",line) 
    return line

#文本切割
def sent2word(line):
    segList = jieba.cut(line,cut_all=False)    
    segSentence = ''
    for word in segList:
        if word != '\t':
            segSentence += word + " "
    return segSentence.strip()


#去除停用词
def stopWord(sourceFile,targetFile,stopkey):
    sourcef = codecs.open(sourceFile, 'r', encoding='utf-8')
    targetf = codecs.open(targetFile, 'w', encoding='utf-8')
    print ('open source file: '+ sourceFile)
    print ('open target file: '+ targetFile)
    lineNum = 1
    line = sourcef.readline()
    while line:
        print ('---processing ',lineNum,' article---')
        sentence = delstopword(line,stopkey)
        #print sentence
        targetf.writelines(sentence + '\n')       
        lineNum = lineNum + 1
        line = sourcef.readline()
    print ('well done.')
    sourcef.close()
    targetf.close()
    

#删除停用词
def delstopword(line,stopkey):
    wordList = line.split(' ')          
    sentence = ''
    for word in wordList:
        word = word.strip()
        if word not in stopkey:
            if word != '\t':
                sentence += word + " "
    return sentence.strip()

#从词向量模型中提取文本特征向量
# 返回特征词向量
def getWordVecs(wordList,model):
    vecs = []
    for word in wordList:
        word = word.replace('\n','')
        #print word
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype='float')
    

# 构建文档词向量 
def buildVecs(filename,model):
    fileVecs = []
    with codecs.open(filename, 'rb', encoding='utf-8') as contents:
        for line in contents:
            logger.info("Start line: " + line)
            wordList = line.split(' ')
            vecs = getWordVecs(wordList,model)
            #print vecs
            #sys.exit()
            # for each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) >0:
                vecsArray = sum(np.array(vecs))/len(vecs) # mean
                #print vecsArray
                #sys.exit()
                fileVecs.append(vecsArray)
    return fileVecs


if __name__ == '__main__':   
    sourceFile = 'weibo.txt'
    targetFile = 'weibo_cut.txt'
    prepareData(sourceFile,targetFile)

    stopkey = [w.strip() for w in codecs.open('stopWord.txt', 'r', encoding='utf-8').readlines()]
    sourceFile = 'weibo_cut.txt'
    targetFile = 'weibo_cut_stopword.txt'
    stopWord(sourceFile,targetFile,stopkey)

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    # load word2vec model
    inp = 'wiki.zh.text.vector'
    model = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)
    
    testInput = buildVecs('weibo_cut_stopword.txt',model)
   

    X = testInput[:]
    X = np.array(X)

    # write in file   
    df_x = pd.DataFrame(X)
    test = pd.concat([df_x],axis = 1)
    #print data
    test.to_csv('weibo.csv')
