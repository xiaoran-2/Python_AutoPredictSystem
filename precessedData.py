
# coding: utf-8

# In[32]:

from bs4 import BeautifulSoup
import numpy as np


import gensim
from gensim import corpora, models, similarities
import re
from gensim.parsing.preprocessing import STOPWORDS
stopwords = STOPWORDS


# fquestion = 'fields.html'
# fanswer = 'evaluator13709346/evaluation2802.html'
# fanswer = 'evaluator1904283/evaluation189.html'
# fanswer = 'evaluator6895718/evaluation1714.html'
# fanswer = 'evaluator8542011/evaluation683.html'
# questionlen = 0
# user_id = 0


# In[159]:

def processedQuestion(q):
    tmp = []
    string = q.get_text()
    que = string[:string.find(':')+1]
    #去电多余的空格，只留一个
    que = ' '.join(que.split())
    
    tmp.append(que)
    lis = q.find_all('li')
    for li in lis:
#         print(str(li)[str(li).find('<li>')+4:str(li).find('.')])
        a = str(li)[str(li).find('<li>')+4:str(li).find('.')]
        a = ' '.join(a.split())
        tmp.append(a)

    return tmp

def findQuestion(texts):
    '''
    找到问题1\2\3\4的位置
    问题下面的div对一个的是答案
    '''
    return texts.get_text().find('Q1')>=0 or texts.get_text().find('Q2')>=0 or texts.get_text().find('Q3')>=0 or texts.get_text().find('Q4')>=0
    

#格式化输入的试卷文件
def processedQuestionFile(filepath):
    f = open(filepath, 'rb')
    htmlstring = f.read()
    #得到所有的div块的文件
    soup = BeautifulSoup(htmlstring)
    
    user_id = soup.title.get_text()
#     print(soup.title)
    
    texts = soup.find_all('div')
#     print("len(texts):",len(texts))
#     print(texts)
    
    Questions_and_Answers = [] #结构 [大问题，子问题，子问题,答案],[大问题，子问题，子问题，答案]...,[大问题，子问题，子问题，答案]
    for i in range(0,len(texts)-1,1):
        if findQuestion(texts[i]): 
#             print(texts[i])
            tmp = processedQuestion(texts[i])
#             print(tmp)
            a = texts[i+1].get_text()
#             print(a)
            a = ' '.join(a.split())
            tmp.append(a)
            Questions_and_Answers.append(tmp)

    # 去掉/n的换行
    for i in range(len(Questions_and_Answers)):
        for j in range(len(Questions_and_Answers[i])):
            try:
                Questions_and_Answers[i][j] = Questions_and_Answers[i][j].replace('\n',' ')
            except Exception:
                Questions_and_Answers[i][j] = []
        
    Questions_and_Answers_s = sorted(Questions_and_Answers)#排序，第一题在前面
    
    Questions_and_Answers = []
    for i in range(len(Questions_and_Answers_s)):
        if len(Questions_and_Answers_s[i][0]) >0 and Questions_and_Answers_s[i][0][0] == 'Q':
            Questions_and_Answers.append(Questions_and_Answers_s[i])
            
    
    
    #得到小问题的个数
    questionlen = 0
    for i in range(len(Questions_and_Answers)):
        questionlen += len(Questions_and_Answers[i]) - 2
        
    
    return Questions_and_Answers,user_id,questionlen
            


# In[160]:

#格式化结果文件,[[问题，结果，评分],...,[问题，结果，评分]]
def processedAnswerFile(filepath,questionlen):
    f = open(filepath, 'rb')
    htmlstring = f.read()

    soup = BeautifulSoup(htmlstring)
    texts = soup.find_all('div') #得到所有的div块的数据
    
#     print(len(texts))
    
    n = min(int(questionlen * 2),34)
    evalution = []
    #去掉最后一个div，最后一个不是评论的答案
    for i in range(0, n-1,2):
        t = texts[i].get_text()
        question = t[:t.find(':')]
        answer = t[t.find(':')+1:].replace("\n"," ")
        
        question = ' '.join(question.split())
        answer = ' '.join(answer.split())
        
#         print(texts[i+1].get_text())
        
        evalution.append([question,answer,int(texts[i+1].get_text())])

    #去掉特殊字符'\xa0'
    for i in range(len(evalution)):
        evalution[i][0] = evalution[i][0].replace("\xa0","")
        evalution[i][1] = evalution[i][1].replace("\xa0","")
    
    return evalution
    



# In[161]:

def countIDF(text,topK = 8):
    '''
    text:字符串，topK根据TF-IDF得到前topk个关键词的词频，用于计算相似度
    return 词频vector
    '''
    from jieba import analyse
    tfidf = analyse.extract_tags

    cipin = {} #统计分词后的词频
    text = text[:text.find('.')]
    fencis = text.split(" ")
    
    #处理特殊字符
    fenci = []
    for i in range(len(fencis)):
        fenci.extend(fencis[i].split("-"))
    
    fencis = fenci
    fenci = []
    for i in range(len(fencis)):
        fenci.extend(fencis[i].split("/"))
    
#     print(fenci)
    #记录每个词频的频率
    for word in fenci:
        if word not in cipin.keys():
            cipin[word] = 0
        cipin[word] += 1

    # 基于tfidf算法抽取前10个关键词，包含每个词项的权重
    keywords = tfidf(text,topK,withWeight=True)
#     print(keywords)
    ans = []
    # keywords.count(keyword)得到keyword的词频
    # help(tfidf)
    # 输出抽取出的关键词
    for keyword in keywords:
        #print(keyword ," ",cipin[keyword[0]])
        if keyword[0] in cipin.keys():
            ans.append(cipin[keyword[0]]) #得到前topk频繁词项的词频的频率

    return keywords, cipin

def cos_sim(a,b):
    '''
    a，b的余弦相似度
    '''
    a = np.array(a)
    b = np.array(b)
        
    #return {"文本的余弦相似度:":np.sum(a*b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))}
    return np.sum(a*b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))
    


# In[ ]:




# In[162]:

def lcs(str_a, str_b):
    lensum = float(len(str_a) + len(str_b))
    #得到一个二维的数组，类似用dp[lena+1][lenb+1],并且初始化为0
    lengths = [[0 for j in range(len(str_b)+1)] for i in range(len(str_a)+1)]

    #enumerate(a)函数： 得到下标i和a[i]
    for i, x in enumerate(str_a):
        for j, y in enumerate(str_b):
            if x == y:
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
    
    return lengths[len(str_a)][len(str_b)]
    
def equals(a,b):
    '''
    如果两个字符串的所有相同单词的个数并 == 单数数-2 ，
    则认为是同一个句子的，不同表达，注意仅在次应用中有用
    '''
    return lcs(a,b) >= min(len(a),len(b)) - 2

def getallAnswers(Questions_and_Answers,evalution):
    '''
    得到所有的答案，作为总的文本计算tfidf
    包括评价文本，答案文本，
    '''
    allAnswers = []
    
    n = len(Questions_and_Answers)
    #整合每一道提的答案和标准的评分答案
    for j in range(n):
        data = Questions_and_Answers[j]
        Answers = []
        Answers.append(data[-1])
        for i in range(1,len(data)-1):
#             print(data[i])
            for k in range(len(evalution)):
                if equals(data[i], evalution[k][0]):
#                     print(evalution[k][0])
                    Answers.append([evalution[k][1],evalution[k][2]])
        
        allAnswers.append(Answers)
        
    return allAnswers




def tokenize(text):
    '''
    处理文本
    '''
    text = text.lower()
    words = re.sub("\W"," ",text).split()
    words = [w for w in words if w not in stopwords]
    return words

# processed_docs = [tokenize(doc) for doc in documents]

def similarityLDA(myAnswer, normalAnswer,source, topK = 12):
    '''
    根据LDA计算相似度，根据评分答案，给出相应的分数
    根据回答问题的答案，得到LDA的topic模型，
    得到评分答案的topic模型，然后在比较两个文本的相似度
    '''
    #回答问题的答案，得到LDA的topic模型
    processed_docs = [tokenize(myAnswer)]
    
#     print(processed_docs)
    dic = corpora.Dictionary(processed_docs) #构造词典  
#----------------------------------------------------------------------------------------------
#    corpus = [dic.doc2bow(text) for text in processed_docs] # 每个text 对应的稀疏向量  
#    tfidf = models.TfidfModel(corpus) #统计tfidf  
#    corpus_tfidf = tfidf[corpus]  #得到每个文本的tfidf向量，稀疏矩阵  
    
    lda = models.LdaModel(id2word = dic, num_topics = topK)   
#-----------------------------------------------------------------------------------------------
#     corpus_lda = lda[corpus_tfidf]
    
    
    # 打印前10个topic的词分布，这儿可以进行输出
#     lda.print_topics(10)
    
    
    #计算normalAnswer的相似度
    test_doc = tokenize(normalAnswer)#新文档进行分词
    doc_bow = dic.doc2bow(test_doc)      #文档转换成bow
    doc_lda = lda[doc_bow]                   #得到新文档的主题分布，
    #输出新文档的主题分布
    #print(doc_lda)
    
    #这里去第一个作为最相关的主题
    topicid = doc_lda[0][0]
    topicSim = doc_lda[0][1]
    
    return source * topicSim
    



# In[163]:

def getSource(myAnswer, normalAnswer,source, topK = 12):
    '''
    myAnswser : 测试的答案
    normalAnswer ：标准打分答案
    topk ：频繁词项的个数
    source : 评分
    '''
#     print("myA",myAnswer)
#     print("normal",normalAnswer)

    keyword0,cinpin0 = countIDF(myAnswer,topK)
    keyword1,cinpin1 = countIDF(normalAnswer,topK)    
#     print("cinpin0",cinpin0)
    
    keywords = set()
    for d in keyword0:
        keywords.add(d[0])

    for d in keyword1:
        keywords.add(d[0])

    # print(keywords)
    tf0 = []
    tf1 = []
    for d in keywords:
    #     print(d)
        if d in cinpin0.keys():
            tf0.append(cinpin0[d] / len(cinpin0))
        else:
            tf0.append(0)
        if d in cinpin1.keys():
            tf1.append(cinpin1[d] / len(cinpin1))
        else:
            tf1.append(0)
#     print(tf0,tf1)
#     print(cos_sim(tf0,tf1) * source)
    return cos_sim(tf0,tf1) * source



# In[164]:

# Questions_and_Answers,user_id,questionlen = processedQuestionFile(fquestion)
# evalution = processedAnswerFile(fanswer)
        


# In[165]:

# allAnswers = getallAnswers(Questions_and_Answers,evalution)


# In[166]:

# len(Questions_and_Answers)


# In[167]:

def getOneSourceAll(Questions_and_Answers,evalution):
    allAnswers = getallAnswers(Questions_and_Answers,evalution)
#     print('allAnswers-------',allAnswers)
    onesourceall = []
    for i in range(len(allAnswers)):
        sourceall = []
        for j in range(1,len(allAnswers[i])):
            sourceall.append(getSource(allAnswers[i][0],allAnswers[i][j][0],allAnswers[i][j][1]))

        onesourceall.append(sourceall)
    return onesourceall


# In[168]:

# onesourceall = getOneSourceAll(Questions_and_Answers,evalution)

# onesourceall


# 得到每个答案的关键次
def getQuestonKeys(filepath):
    Questions_and_Answers,user_id,questionlen = processedQuestionFile(filepath)

    QuestonKeys = []
    topK = 12 #可以修改关键词的个数
    for i in range(len(Questions_and_Answers)):
        myAnswer = Questions_and_Answers[i][-1]
        keyword0,cinpin0 = countIDF(myAnswer,topK)
        
        keywords = [keyword0[i][0] for i in range(len(keyword0))]
        QuestonKeys.append(keywords)
    
    return QuestonKeys



#得到每个问题答案的LDA的矩阵
def getQuestonLDAMat(filepath):
    Questions_and_Answers,user_id,questionlen = processedQuestionFile(filepath)
    
    QuestonLDAMat = []
    for i in range(len(Questions_and_Answers)):
        myAnswer = Questions_and_Answers[i][-1]
        #回答问题的答案，得到LDA的topic模型
        processed_docs = [tokenize(myAnswer)]
        # print(processed_docs)
        dic = corpora.Dictionary(processed_docs) #构造词典  
#        corpus = [dic.doc2bow(text) for text in processed_docs] # 每个text 对应的稀疏向量  
#        tfidf = models.TfidfModel(corpus) #统计tfidf  
#        corpus_tfidf = tfidf[corpus]  #得到每个文本的tfidf向量，稀疏矩阵  

        lda = models.LdaModel(id2word = dic, num_topics = 12)   
    
        QuestonLDAMat.append(lda.print_topics())
    
    return QuestonLDAMat



# In[169]:

#集成一个文件进行测试：
def uniteAll(filelist):
    
    if len(filelist)<2 :
        return None,None
    
    fquestion = filelist[0]
    sourcelist = []
    
    for i in range(1,len(filelist)):
        fanswer = filelist[i]
        Questions_and_Answers,user_id,questionlen = processedQuestionFile(fquestion)
#         print(Questions_and_Answers,questionlen)
        
        evalution = processedAnswerFile(fanswer,questionlen)
#         print(evalution)
        user_id = user_id[user_id.find(':')+1:user_id.find(')')].strip()
        onesourceall = getOneSourceAll(Questions_and_Answers,evalution)
        
#         print(onesourceall)
        
        sourcelist.append(onesourceall)
#         print(onesourceall)
    
    source = []
    try:
        for k in range(min(len(sourcelist[0]),4)):
            tmp = sourcelist[0][k]
            for i in range(1,len(sourcelist)):
                tmp =[tmp[j] + sourcelist[i][k][j] for j in range(len(sourcelist[i][k]))]

            source.append(tmp)
    except Exception:
        pass

    source4 = []
    for i in range(len(source)):
        source4.append(np.sum(np.array(source[i]))/len(source[i]))
        
    return user_id,source4




# In[170]:
'''
filelist = ['/home/xiaoran/tmpfiles/tmp/submission441/fields.html',
 '/home/xiaoran/tmpfiles/tmp/submission441/evaluator13709346/evaluation2802.html',
 '/home/xiaoran/tmpfiles/tmp/submission441/evaluator6895718/evaluation1714.html',
 '/home/xiaoran/tmpfiles/tmp/submission441/evaluator9905753/evaluation2022.html',
 '/home/xiaoran/tmpfiles/tmp/submission441/evaluator8542011/evaluation683.html',
 '/home/xiaoran/tmpfiles/tmp/submission441/evaluator1904283/evaluation189.html']


filelist1 = ['1/assessment5/submitter4259923/submission388/fields.html',
 '1/assessment5/submitter4259923/submission388/evaluator9247166/evaluation245.html',
 '1/assessment5/submitter4259923/submission388/evaluator13826915/evaluation1029.html',
 '1/assessment5/submitter4259923/submission388/evaluator5456952/evaluation1752.html',
 '1/assessment5/submitter4259923/submission388/evaluator1755159/evaluation2330.html']
'''

# In[173]:

# user_id,source = uniteAll(filelist)


# In[174]:

# user_id,source


# In[ ]:


