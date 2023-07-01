import os
import random
import numpy as np
import jieba
import math

class LDA_Model:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.data_txt, self.files = self.read_novel(corpus_path)
        self.doc_count = []
        self.doc_fre = []
        self.topic_count = []
        self.topic_all = []
        self.topic_fre_list = [{} for _ in range(len(self.data_txt))]
        self.doc_pro = []

    def read_novel(self, path):
        content = []
        names = os.listdir(path)
        for name in names:
            con_temp = []
            novel_name = path + '\\' + name
            with open(novel_name, 'r', encoding='ANSI') as f:
                con = f.read()
                con = self.content_deal(con)
                con = jieba.lcut(con)
                con_list = list(con)
                pos = int(len(con) // 13)
                for i in range(13):
                    con_temp = con_temp + con_list[i * pos:i * pos + 500]
                content.append(con_temp)
            f.close()
        return content, names

    def content_deal(self, content):
        ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com',
              '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库', '\u3000', '\n', '。', '？', '！', '，',
              '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......', '『', '』', '（',
              '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
        for a in ad:
            content = content.replace(a, '')
        return content

    def random_init_model(self):
        for data in self.data_txt:
            topic = []
            docfre = {}
            for word in data:
                a = random.randint(0, len(self.data_txt) - 1)
                topic.append(a)
                if '\u4e00' <= word <= '\u9fa5':
                    self.topic_count.append(a)
                    docfre[a] = docfre.get(a, 0) + 1  # 统计每篇文章的词频
                    self.topic_fre_list[a][word] = self.topic_fre_list[a].get(word, 0) + 1
            self.topic_all.append(topic)
            docfre = list(dict(sorted(docfre.items(), key=lambda x: x[0], reverse=False)).values())
            self.doc_fre.append(docfre)
            self.doc_count.append(sum(docfre))
        self.topic_count = list(dict(sorted(dict(zip(range(len(self.data_txt)), self.topic_count)).items(),
                                            key=lambda x: x[0], reverse=False)).values())
        self.doc_fre = np.array(self.doc_fre)
        self.topic_count = np.array(self.topic_count)
        self.doc_count = np.array(self.doc_count)
        for i in range(len(self.data_txt)):
            doc_up = np.divide(self.doc_fre[i], self.doc_count[i])
            self.doc_pro.append(doc_up)
        self.doc_pro = np.array(self.doc_pro)

    def model_training(self):
        stop = 0
        loopcount = 1
        while stop == 0:
            for i, data in enumerate(self.data_txt):
                top = self.topic_all[i]
                for w in range(len(data)):
                    word = data[w]
                    word_upper_bound = False
                    if '\u4e00' <= word <= '\u9fa5':
                        word_upper_bound = True
                        pro = self.doc_pro[i] *
                              np.array([self.topic_fre_list[j].get(word, 0) for j in range(len(self.data_txt))]) /
                              self.topic_count  # 计算每篇文章选中各个topic的概率乘以该词语在每个topic中出现的概率，得到该词出现的概率向量
                        m = np.argmax(pro)
                        self.doc_fre[i][top[w]] -= 1
                        self.doc_fre[i][m] += 1
                        self.topic_count[top[w]] -= 1
                        self.topic_count[m] += 1
                        self.topic_fre_list[top[w]][word] = self.topic_fre_list[top[w]].get(word, 0) - 1
                        self.topic_fre_list[m][word] = self.topic_fre_list[m].get(word, 0) + 1
                        top[w] = m
                self.topic_all[i] = top
            if loopcount == 1:
                doc_pronew = np.array([np.divide(self.doc_fre[i], self.doc_count[i]) for i in range(len(self.data_txt))])
            else:
                doc_pronew = np.array([np.divide(self.doc_fre[i], self.doc_count[i]) for i in range(len(self.data_txt))])
            if (doc_pronew == self.doc_pro).all():
                stop = 1
            else:
                self.doc_pro = doc_pronew.copy()
            loopcount += 1
        return self.doc_pro, loopcount

if __name__ == '__main__':
    lda = LDA_Model("金庸小说集")
    lda.random_init_model()
    doc_pro, loopcount = lda.model_training()
    print("模型训练完毕，迭代次数:", loopcount)
    test_lda = LDA_Model("金庸小说集")
    test_lda.random_init_model()
    test_doc_pro, test_loopcount = test_lda.model_training()
    print("测试集测试完毕，迭代次数:", test_loopcount)