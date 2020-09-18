# 스코어 기준과 loss 기준. lb 점수가 cv score와 비교했을 때 굉장히
# consistent해서 cv score를 기준으로 합니다.

def pick_best_score(result1, result2):
    if result1['best_score'] < result2['best_score']:
        return result2
    else:
        return result1
    
def pick_best_loss(result1, result2):
    if result1['best_loss'] < result2['best_loss']:
        return result1
    else:
        return result2