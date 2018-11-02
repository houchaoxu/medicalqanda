import jieba
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import jieba.posseg as pseg
import numpy as np
from medicalQA.knowledge.KnowledgeModels import question_set

class Common(object):

    def __init__(self):
        print('这里就是构造函数的开始')
        self.vector_model_path = 'medicalQA/common/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
        self.vector_model = gensim.models.KeyedVectors.load_word2vec_format(self.vector_model_path,binary=True,limit=100000)
        # 加载停用词表
        self.stopwords = [line.strip() for line in open('medicalQA/common/停用词.txt', 'r', encoding='utf-8').readlines()]
        # 加载自定义词典
        self.entities = [line.strip().split()[0] for line in open('medicalQA/common/named_entity.txt', 'r', encoding='utf-8').readlines()]
        self.properties = [line.strip().split()[0] for line in open('medicalQA/common/named_property.txt', 'r', encoding='utf-8').readlines()]
        named_entity_file = open('medicalQA/common/named_entity.txt', 'r', encoding='utf-8')
        named_property_file = open('medicalQA/common/named_property.txt', 'r', encoding='utf-8')
        jieba.load_userdict(named_entity_file)
        jieba.load_userdict(named_property_file)
        self.question_set = question_set
        self.question_list = []
        self.goal_question_vector(self.question_set)

    def cut_stop(self, question):

        # 带词性标注的分词
        answer = pseg.cut(question, HMM=False)
        result = []
        word_filter = ['n','m','Ng','nr','ns','nt','nz','q','s','t','vn','x']
        for w in answer:
            print(w.word, w.flag)
            if w.word not in self.stopwords and w.flag in word_filter:
                result.append(w.word)
        return result

    def goal_question_vector(self, question_set):
        for index, question in enumerate(question_set):
            words = jieba.cut(question,HMM=False)
            self.question_list.append(self.word2vector(words, sentence=True))


    def sentence_similarity(self, question):
        # 对句子进行分词
        words_list = jieba.cut(question, HMM=False)
        # 筛选出实体or标签
        words = []
        query_entity = ''
        query_property = ''
        for word in words_list:
            words.append(word)
            if (word) in self.entities:
                query_entity = word
            if (word) in self.properties:
                query_property = word
        print('query_entity', query_entity)
        # 筛选出两个关键词
        key_words = self.cut_stop(question)
        print('keywords', key_words)
        # 计算当前语句句向量
        # 替换掉当前句子中的实体和属性
        for index, word in enumerate(words):
            print(word, query_entity)
            if word in key_words or word == query_entity:
                if word == query_entity:
                    words[index] = '实体'
                else:
                    words[index] = '属性'
        # 当前句子的句向量
        print('wordlist ', [word for word in words])
        goal_vector = self.word2vector(words,sentence=True)
        # 在question_vector中查询最接近goal_vector的句子
        def cos_similar(embedding1, embedding2):
            embedding1 = np.matrix(embedding1)
            embedding2 = np.matrix(embedding2)
            num = float(embedding1 * embedding2.T)
            denom = np.linalg.norm(embedding1)*np.linalg.norm(embedding2)
            cos = num/denom
            return cos
        compare_list = []
        for vector in self.question_list:
            compare_list.append(cos_similar(goal_vector,vector))
        maxval = np.array(compare_list).max()
        query_index = compare_list.index(maxval)
        print('得到的问句是：',query_index)
        if key_words[0] == query_entity:
            key_words.remove(key_words[0])
        return query_entity, ' '.join(key_words), query_index

    def sentence_similarity_medical(self, question):
        # 对句子进行分词
        words_list = jieba.cut(question, HMM=False)
        # 筛选出实体or标签
        words = []
        query_entity = ''
        for word in words_list:
            words.append(word)
            if (word) in self.entities:
                query_entity = word
            if (word) in self.properties:
                query_property = word
        print('query_entity', query_entity)
        # 筛选出两个关键词
        key_words = self.cut_stop(question)
        print('keywords', key_words)
        # 计算当前语句句向量
        # 替换掉当前句子中的实体和属性
        for index, word in enumerate(words):
            print(word, query_entity)
            if word in key_words or word == query_entity:
                if word == query_entity:
                    words[index] = '实体'
                else:
                    words[index] = '属性'
        # 当前句子的句向量
        print('wordlist ', [word for word in words])
        goal_vector = self.word2vector(words,sentence=True)
        # 在question_vector中查询最接近goal_vector的句子
        def cos_similar(embedding1, embedding2):
            embedding1 = np.matrix(embedding1)
            embedding2 = np.matrix(embedding2)
            num = float(embedding1 * embedding2.T)
            denom = np.linalg.norm(embedding1)*np.linalg.norm(embedding2)
            cos = num/denom
            return cos
        compare_list = []
        for vector in self.question_list:
            compare_list.append(cos_similar(goal_vector,vector))
        maxval = np.array(compare_list).max()
        query_index = compare_list.index(maxval)
        print('得到的问句是：',query_index)
        return query_entity, query_index

    def word2vector(self, word_list, sentence=False):
        vectors = []
        result = []
        for word in word_list:
            vectors.append(self.vector_model[word])
        if sentence == False:
            for vector in vectors:
                result.append(vector)
            return result
        else:
            v_zero = np.zeros(64)
            for vector in vectors:
                v_zero += vector
            return v_zero

common = Common()

