##새싹 3주차_허깅페이스 파이프라인 실습 3-1 감정분석 복붙...
from transformers import pipeline

sentiment_analysis = pipeline("sentiment-analysis", model="Hyeonuuuuuu/bert-sentiment-ko")
  
def run_sentiment(text: str) -> str:
    
    result = sentiment_analysis(text)
    output = result[0]['label']
    
    if output == "LABEL_1": 
        return "긍정인듯?"
    else:
        return "부정인가?"