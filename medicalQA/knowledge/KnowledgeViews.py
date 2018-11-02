import json
from django.http import HttpResponse
from django.shortcuts import render
from medicalQA.common.CommonFunctions import common
from medicalQA.knowledge import KnowledgeModels
from py2neo import Graph

def toJson(data, error, status):
    dict = {}
    dict['status'] = status
    dict['msg'] = error
    dict['data'] = data
    return json.dumps(dict, ensure_ascii=False)

def knowledge_qa(request):
    goal = request.GET.get('goal', 'None')
    status = 0
    error = ' '
    if goal == 'qa':
        sentence = request.GET.get('sentence', 'None')
        if sentence == 'None':
            status = 1
            error = '问句不能为空'
            data = ' '
            response = toJson(data, error, status)
            return HttpResponse(response)
        else:
            # 先对问句进行分词
            first, second, number = common.sentence_similarity(sentence)
            Qtype = KnowledgeModels.question_set[number]
            graph = Graph("localhost:7474", username="neo4j", password="1039")
            sql = graph.evaluate("match(e:Qands) where e.Qtype={s} return e.sql", s=Qtype)
            sql = sql.replace('$', first)
            sql = sql.replace('#', second)
            sql = sql.replace('%', second)
            df = graph.evaluate(sql)
            data = ' '.join([str(number), first, second, sql])
            response = toJson(data, error, status)
        return HttpResponse(df)
p_id = 1
def medical_qa(request):
    global p_id
    question = request.GET.get('question', 'None')
    if question == 'None':
        status = 1
        error = '问句不能为空'
        data = ' '
        response = toJson(data, error, status)
        return HttpResponse(response)
    else:
        # 先对问句进行分词
        first, number = common.sentence_similarity_medical(question)
        Qtype = KnowledgeModels.question_set[number]
        graph = Graph("localhost:7474", username="neo4j", password="1039")
        sql = graph.evaluate("match(e:Qands) where e.Qtype={s} return e.sql", s=Qtype)
        sql = sql.replace('*', (str)(p_id))
        sql = sql.replace('$', first)
        df = graph.evaluate(sql)
        flag = request.POST.get('input')
        if flag == None:
            return render(request, 'medicalQA/medicalQA.html', {'goal': df})
        else:
            df, p_id = f(p_id, flag, first)
            return render(request, 'medicalQA/medicalQA.html', {'goal': df})
        # p_id = Graph.evaluate("match (e)-[r:关系]->(s) where s.included={s} and s.id=1 return s.id", s = first)


def f(p_id, flag, str):
    graph = Graph("localhost:7474", username="neo4j", password="1039")
    cql = "match(e)-[r:关系]->(c) where e.id={} and r.type='".format(p_id)+"{}'".format(flag)+" and e.included='{}'".format(str)+" return c.name"
    p_id = graph.evaluate("match(e)-[r:关系]->(c) where e.id={} and r.type='".format(p_id)+"{}'".format(flag)+" and e.included='{}'".format(str)+" return c.id")
    str = graph.evaluate(cql)
    if str == None:
        str = '谢谢使用'
        p_id = 1
        return str, p_id
    else:
        return str, p_id