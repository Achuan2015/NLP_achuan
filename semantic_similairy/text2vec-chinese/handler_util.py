from hrtps_file_manager import HRTPSFileManager
import difflib

def get_data_manager():
    # 数据管理中心
    allow_id = "zhangc-df04e"
    data_manager = HRTPSFileManager(allow=allow_id)
    return data_manager

def badword_filter(text, badwords):
    is_contain = False
    for word in badwords:
        if word in text:
            is_contain = True
            break
    return is_contain


def calculate_edit_score(corpus, candidates, threshold, target_intent, edit_weight=None, enhanced_match=0):
    result = set()
    result_verbose = {}
    result_ratio_verbose = {}
    
    VECTOR_WEIGHT = 0.6
    
    EDIT_WEIGHT = edit_weight if edit_weight is not None else 0.4
    INSTANCE_QUERY_LENGTH = 20
    EDIT_THRESHOLD = 0.12
    MAX_SCORE = 0.99
    # 单个corpus 编辑距离最高的分数
    EDIT_MATCH_SCORE = 0.9

    if not candidates:
        return result, result_verbose, result_ratio_verbose

    # 记录原始的综合得分score 以及 ratio 得分
    temp_origin_verbose = {}
    temp_ratio_verbose = {}
    for intent_id, items in candidates.items():
        if intent_id in target_intent:
            cur_threshold = target_intent[intent_id]
            for item in items:
                content = item[1]
                score = item[2]
                sm = difflib.SequenceMatcher(None, str(content), str(corpus))
                ratio = sm.ratio()
                if ratio > 0:
                    score = VECTOR_WEIGHT * item[2] + EDIT_WEIGHT * ratio

                    temp_origin_verbose[intent_id] = score
                    temp_ratio_verbose[intent_id] = ratio

                    if intent_id in target_intent and len(target_intent) > 1 and len(str(corpus)) >= INSTANCE_QUERY_LENGTH and enhanced_match == 1:
                        if ratio >= EDIT_THRESHOLD:
                            if score < MAX_SCORE:
                                print(f'intent_id:{intent_id} 触发加分')
                                score = min(score * 1.25, 0.99)

                    if score >= cur_threshold:
                        result.add(intent_id)
                        result_verbose[intent_id] = score
                        result_ratio_verbose[intent_id] = ratio
    # 增加规则判断 编辑 距离的影响
    if len(target_intent) > 1:
        ratio_scores = sorted(list(temp_ratio_verbose.values()),reverse=-1)
        print(f'触发新增规则 ratio_scores:{ratio_scores}')
        if len(ratio_scores) > 1 and ratio_scores[0] >= EDIT_MATCH_SCORE and ratio_scores[0] - ratio_scores[1] > 0.3:
            # 满足条件，重置 intent_id 对应的分数
            result = set()
            result_verbose = {}
            result_ratio_verbose = {}
            for intent_id, temp_score in temp_origin_verbose.items():
                if intent_id in target_intent:
                    cur_threshold = target_intent[intent_id]
                    if temp_score >= cur_threshold:
                        result.add(intent_id)                
                        result_verbose[intent_id] = temp_score
                        result_ratio_verbose[intent_id] = temp_ratio_verbose[intent_id]
    
    return result, result_verbose, result_ratio_verbose


def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e: #转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


# if __name__ == '__main__':
#     corpus = '你好，我是银行的工作人员，尾号1234'
#     candidates = {'10392428478': [('001', '你好', 0.73), ('002', '我是银行的工作人员', 0.81), ('003', '尾号1234', 0.74)]}
#     threshold=0.7
#     result = calculate_edit_score(corpus, candidates, threshold)
#     print(result)
