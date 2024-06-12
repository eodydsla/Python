import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_article_details(link):
    article_response = requests.get(link)
    article_soup = BeautifulSoup(article_response.content, 'html.parser')
    
    content_tag = article_soup.find(id='anchorTop')
    content = content_tag.text.strip() if content_tag else "No Content"
    return content

# 전체 페이지 수 설정
total_pages = 2

# 데이터 저장할 리스트
data = []

for page in range(1, total_pages + 1):
    print(f"{page}페이지 작업중...")
    
    # 페이지 URL 설정
    url = f"https://www.waterjournal.co.kr/news/articleList.html?page={page}&total=70128&box_idxno=&view_type=sm"
    
    # 웹 페이지 요청
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 기사 목록 추출
    articles = soup.find_all('li')
    
    # 각 기사에서 제목, 날짜, 내용을 추출
    for article in articles:
        title_tag = article.find('h4', class_='titles')
        if not title_tag:
            continue
        
        title = title_tag.text.strip() if title_tag else "No Title"
        link = title_tag.find('a')['href'] if title_tag.find('a') else None
        link = "https://www.waterjournal.co.kr" + link if link else None

        date_tag = article.find('span', class_='byline').find_all('em')[-1]
        date = date_tag.text.strip() if date_tag else "No Date"
        
        # 개별 기사 페이지 요청 및 내용 추출
        if link:
            content = get_article_details(link)
            data.append({'제목': title, '날짜': date, '내용': content, 'URL': link})
        
        # 요청 사이에 1초 대기
        time.sleep(1)
    
    print(f"{page}페이지 작업 완료.")

# 데이터프레임으로 변환
df = pd.DataFrame(data)

# 결과 출력
#print(df)

# 데이터프레임을 CSV 파일로 저장
#df.to_csv("waterjournal_articles.csv", index=False, encoding='utf-8-sig')
