from django.http import HttpResponse
from medicalQA.common.CommonFunctions import common
import json

def toJson(data,error,status):
    dict = {}
    dict['status'] = status
    dict['msg'] = error
    dict['data'] = data
    return json.dumps(dict, ensure_ascii=False)

def CommonInference(request):
    goal = request.GET.get('goal', 'None')

    if goal == 'fc':
        sentence = request.GET.get('sentence', 'None')
        status = 0
        error = ' '
        cut_result = common.cut_stop(sentence)
        print(cut_result)
        data = ' '.join(cut_result)
        print(data)
        print(toJson(data, error, status))
        response = toJson(data, error, status)
        return HttpResponse(response)

    if goal == 'w2v':
        words = request.GET.get('words', 'None')
        sentence = request.GET.get('sentence', 'false')
        words = words.strip().split()
        print(words)
        status = 0
        error = ' '
        if sentence == 'false':
            result = common.word2vector(words, sentence=False)
            data = str(list(result))
        elif sentence == 'true':
            result = common.word2vector(words, sentence=True)
            data = str(list(result))
        else:
            data = ' '
            status = 1
            error = 'sentence参数必须为false与true之一'
        return HttpResponse(toJson(data, error, status))