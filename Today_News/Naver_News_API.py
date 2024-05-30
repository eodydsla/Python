import os
import urllib.request
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import openai
from tqdm import tqdm
from openai import OpenAI
import schedule
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import re

os.environ["OPENAI_API_KEY"] = 
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key = api_key)

def GPTclass(x, y):
    content = x + y
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = [
            {"role": "system", "content":"환경(기후)정책 및 현황 관련이 높은 뉴스이면 1, 그이외거나 사기업 기사는 0 으로 분류, 반드시 1,0값만 출력"},
            {"role": "user",  "content": f"기사:{content}"}
        ]
    )
    return response.choices[0].message.content

def remove_specific_tags(text):
    text = text.replace('<b>', '').replace('</b>', '')
    return text

def TitleClean(title, description):
    content = f"{title} {description}"
    RMWords = ["<b>", "</b>", "&quot", ";"]
    for x in RMWords:
        content = content.replace(x, " ")
    return content

# Function to preprocess titles (simple tokenization and normalization)
def preprocess_title(title):
    # Normalize by lowecasing and splitting by spaces (simple tokenization)
    return set(re.split(r'\W+', title.lower()))

# 기사 제목을 임베딩하기 위한 함수
def get_embedding(text):
    return client.embeddings.create(input = text, model='text-embedding-3-large').data[0].embedding

def get_media_name(link,media_df):
    for idx, row in media_df.iterrows():
        if row['주소 키워드'] in link:
            return row['언론사 이름']
    return '기타' 
    
def news_collector() :
    client_id = ""
    client_secret ="" 
    
    # == 한국환경연구원 기사 == 
    keywords = ['"한국환경연구원"']
    date_format = "%a, %d %b %Y %H:%M:%S %z"
    kei_df = pd.DataFrame()
    freq_list = []
    for key in keywords:
        search_query = f"{key}"
        encText = urllib.parse.quote(search_query)
        result_df = pd.DataFrame()
        df_list = []

        # 검색 결과를 저장할 빈 리스트 생성
        news_list = []
        for start in range(1 , 1001, 100): # 100개씩 10번 호출
            url = f"https://openapi.naver.com/v1/search/news?query={encText}&start={start}&display=100&sort=sim"  # JSON 결과= 
            request = urllib.request.Request(url)
            request.add_header("X-Naver-Client-Id",client_id)
            request.add_header("X-Naver-Client-Secret",client_secret)
            response = urllib.request.urlopen(request)
            rescode = response.getcode()
            if(rescode==200):
                response_body = response.read().decode('utf-8')
                news_list.append(response_body)
            else:
                print("Error Code:" + rescode)

        # 검색결과를 Dataframe으로 변환
        df_list = []
        for news_json in news_list:
            news_data = json.loads(news_json)
            items = news_data.get('items', [])
            df = pd.DataFrame(items)
            df_list.append(df)

        result_df = pd.concat(df_list)

        # DataFrame을 csv로 저장
        result_df['Key'] = key
        result_df['pubDate'] = pd.to_datetime(result_df['pubDate'], format=date_format)
        now = datetime.now().astimezone()
        today_7am = now.replace(hour=7, minute=0, second=0, microsecond=0)
        one_day_ago_7am = today_7am - timedelta(days=1)

        # 조건에 맞는 데이터 필터링
        result_df = result_df[(result_df['pubDate'] >= one_day_ago_7am) & (result_df['pubDate'] <= today_7am)]
        result_df['pubDate'] = result_df['pubDate'].dt.strftime('%Y-%m-%d %H:%M:%S %z')
        
        fn = "KEI_" + now.strftime("%Y%m%d_%H") + ".xlsx"
        result_df.to_excel("/media/Data/오늘의_환경뉴스/" + fn)
        time.sleep(2)     
              
    keywords = ['UN', 'OECD', 'IPCC', 'UNFCCC', '환경경제', '친환경', '녹색전환', '녹색경제', '지속가능성', '지속가능개발', 'SDGs', 'SDG', '녹색전환', '녹색산업', 'ESG', '환경규제', '환경가치', '녹색금융', '녹색성장', '기후예산', '탄소국경', '환경재정', '환경금융', '환경예산', '국제환경', '지속가능발전', '지속가능사회', '환경사회', '에너지전환', '환경거버넌스', '환경갈등', '환경약자', '시민환경', '환경 시민의식', '녹색시민', '환경교육', '환경정의', '국제개발협력', '국제협력', '국제기구', '국제환경', '환경협상', '환경ODA', '그린ODA', '플라스틱 협약', '자원순환', '자원 전주기', '플라스틱', '배터리', '폐패널', '폐기물', '폐자원', '전기차', '순환도시', '자원순환형 산업단지', '재생원료', '폐열', '부산물 순환이용', '재활용', '리사이클링', '재사용', '제로웨이스트', '일회용품', '환경보건', '환경유해인자', '화확물질', '위해성평가', '건강영향', '환경성질환', '환경보건 수용체', '환경매체', '환경보건 민감계층', '환경보건 평가', '환경보건 모니터링', '환경보건 안전관리', '녹색화학', '생활환경', '요리매연', '온실가스 감축', '온실가스 배출', 'NDC', '탄소중립', '탄소시장', '탄소가격', '기후대응기금', '온실가스 예산', '녹색기술', '기후기술', '신재생에너지', '에너지절감', '기후위기', '기후위기 적응', '쓰레기', '기후변화 적응', '기후변화 영향', '기후변화 피해', '기후변화 취약성', '취약성 평가', '기후변화 위험', '기후위험', '기후리스크', '기후탄력성', '기후위기 취약계층', '기후위기 취약지역', '산업계 적응', '공공기관 적응', '기후적응 교육', '기후적응 협력', '기후적응 정보', '기후적응 주류화', '대기환경', '대기정책', '미세먼지', '오존', '대기오염', '대기오염 교통', '교통환경', '대기오염물질',
                '대기질', '대기배출시설', '이동오염원', '무공해', '온실가스','전기차', '물환경', '물관리', '통합물관리', '물순환', '물산업', '물재해', '물위기', '물부족', '집중호우', '폭우', '홍수', '침수', '도시침수', '가뭄', '폐수', '가축분뇨', '비점오염원', '수량', '수질', '지하수', '토양', '물공급', '수자원', '수리수문', '댐', '하천', '녹조', '치수', '수생태계', '해수담수화', '탄소중립도시', '국토환경', '국토환경정책', '도시환경', '환경계획', '환경정책', '환경관리', '공간조성', '국토공간정보', '환경공간정보', '공간환경정보', '국토환경정보', '스마트 공간환경', '녹색국토', '국토생태', '탄소흡수원', '훼손지', '녹색복원', '자연환경', '자연환경복원', '기후안전', '국토도시', '기후탄력성', '기후탄력도시', '도시전환', '쇠퇴지역', '소멸', '균형발전', '국토생태', '녹색복원', '문화경과', '리빙랩', '그린SOC', '생물다양성', '탄소흡수원', '생태계', '생태계서비스', '자연환경', '자연자원', '자연자산', '자연생태', '동식물', '동물', '식물', '외래생물', '멸종위기', '지질공원', '국립공원', '자연기반해법', '습지', '해양환경', '해양보호', '연안', '하구', '블루카본', '해상풍력', '해양 신재생에너지', '탄소포집', '그린수소', '해양오염', '해양쓰레기', '해양 오염물질 유출사고', '환경평가', '환경성 평가', '환경평가 모니터링', '전략환경영향평가', '환경영향평가', '기후변화영향평가', '소규모영향평가', '사후환경영향조사', '환경성 검토', '환경영향평가 협의', '환경영향평가협의회', '디지털 환경평가', '환경평가 모니터링', '환경매체', '주민의견수렴', '소음', '진동', '기후적응','기후변화','환경세','탄소세','환경부','탄소저장','폭염','배출량','환경오염','환경보전','환경보호']
    keywords = list(set(keywords))
    keywords.sort()
    print(len(keywords))
    
    date_format = "%a, %d %b %Y %H:%M:%S %z"
    base_key = '환경'
    #base_key = ''
    final_df = pd.DataFrame()
    freq_list = []
    for key in keywords:
        search_query_base = base_key
        search_query = f"{search_query_base} {key}"
        #search_query = f"{key}"

        encText = urllib.parse.quote(search_query)

        result_df = pd.DataFrame()
        df_list = []

        # 검색 결과를 저장할 빈 리스트 생성
        news_list = []
        for start in range(1 , 101, 100): # 100개씩 10번 호출
            url = f"https://openapi.naver.com/v1/search/news?query={encText}&start={start}&display=100&sort=sim"  # JSON 결과= 
            request = urllib.request.Request(url)
            request.add_header("X-Naver-Client-Id",client_id)
            request.add_header("X-Naver-Client-Secret",client_secret)
            response = urllib.request.urlopen(request)
            rescode = response.getcode()
            if(rescode==200):
                response_body = response.read().decode('utf-8')
                news_list.append(response_body)
            else:
                print("Error Code:" + rescode)

        # 검색결과를 Dataframe으로 변환
        df_list = []
        for news_json in news_list:
            news_data = json.loads(news_json)
            items = news_data.get('items', [])
            df = pd.DataFrame(items)
            df_list.append(df)

        result_df = pd.concat(df_list)

        # DataFrame을 csv로 저장
        result_df['Key'] = key
        fn = f"{search_query}_news.xlsx"
        result_df['pubDate'] = pd.to_datetime(result_df['pubDate'], format=date_format)
        now = datetime.now().astimezone()
        today_7am = now.replace(hour=7, minute=0, second=0, microsecond=0)
        one_day_ago_7am = today_7am - timedelta(days=1)

        # 조건에 맞는 데이터 필터링
        result_df = result_df[(result_df['pubDate'] >= one_day_ago_7am) & (result_df['pubDate'] <= today_7am)]
        result_df['pubDate'] = result_df['pubDate'].dt.strftime('%Y-%m-%d %H:%M:%S %z')
        result_df.to_excel("./result2/" + fn, index=False)
        freq_list.append([key, result_df.shape[0]])
        print(f"{fn} 파일이 저장되었습니다")
        final_df = pd.concat([final_df,result_df])
        time.sleep(2)        

    final_df = final_df.reset_index(drop=True)
    df1 = final_df[final_df['link'].str.contains('news.naver.com')]
    df2 = final_df[~final_df['link'].str.contains('news.naver.com')]
    final_df = pd.concat([df1, df2]).reset_index(drop=True)
    final_df.to_excel("./final_df.xlsx")
    final_df = pd.concat([df1, df2]).reset_index(drop=True)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(final_df['title'])

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 유사한 제목의 행 제거 (중복된 첫 번째 행만 남김)
    threshold = 0.5  # 유사도를 판단하는 임계값 설정
    drop_idx = set()
    content_aggregate = {}

    # 유사도 계산 및 내용 합치기
    for i in range(len(final_df)):
        if i in drop_idx:
            continue
        content_aggregate[i] = final_df.loc[i, 'Key']
        for j in range(i + 1, len(final_df)):
            if cosine_sim[i, j] > threshold:
                content_aggregate[i] += "," + final_df.loc[j, 'Key']  # 중복 내용을 세미콜론으로 구분하여 추가
                drop_idx.add(j)

    # 중복 제거 및 업데이트된 컨텐츠 열 반영
    final_df = final_df.drop(list(drop_idx)).reset_index(drop=True)
    content_aggregate = list(content_aggregate.values())
    final_df['Key'] = content_aggregate  # 업데이트된 내용 반영
    final_df['Key'] = final_df['Key'].apply(lambda x: ','.join(sorted(set(x.split(',')))))

    final_df['title'] = final_df['title'].map(remove_specific_tags)
    final_df['description'] = final_df['description'].map(remove_specific_tags)

    news = final_df
    news['class'] = [GPTclass(x, y) for x, y in tqdm(zip(news['title'], news['description']), total=len(news),leave=True)]
    news['class'] = news['class'].astype(str)
    news = news[news['class'] == "1"]
    news = news.reset_index(drop=True)
    #news.drop('Unnamed: 0', axis=1, inplace=True)
    news.drop('class', axis=1, inplace=True)
    fn = now.strftime("%Y%m%d_%H") + "2.xlsx"
    news.to_excel("/media/Data/오늘의_환경뉴스/" + fn)
    
    # 임베딩을 활용한 유사기사 제거 및 분류 수행
    
    news_relevant = news
    embeddings_new = np.array([get_embedding(x + y) for x, y in tqdm(zip(news_relevant['title'], news_relevant['description']), total=len(news_relevant), leave=True)] )
    
    cos_dist = pairwise_distances(embeddings_new, embeddings_new, metric='cosine')
    cos_sim = 1 - cos_dist
    np.fill_diagonal(cos_sim, -np.inf)
    indices_to_remove = list(np.where(np.triu(cos_sim) > 0.8)[1])

    embeddings_process = [item for idx, item in enumerate(embeddings_new) if idx not in indices_to_remove]
    news_wo_dup = news_relevant.drop(indices_to_remove).reset_index(drop=True)
    
    news_ref = pd.read_excel("오늘의환경뉴스취합_4월.xlsx")
    embeddings_ref = np.loadtxt("news_reference_embeddings.txt", dtype=float)

    Classify_news = pairwise_distances(embeddings_ref, embeddings_process, metric='cosine')
    Class_index   = np.argmin(Classify_news, axis=0)                 
    Class_column = news_ref['분야'][Class_index]
    
    news_wo_dup['category'] = np.full(len(news_wo_dup), Class_column)   

    # 언론사 이름 매핑
    media_df = pd.read_excel("./언론사주소.xlsx")
    media_df['주소 키워드'] = media_df['주소 키워드']
    
    news_wo_dup['언론사 이름'] = news_wo_dup['originallink'].apply(get_media_name,args=(media_df,))

    fn = now.strftime("%Y%m%d_%H") + "_최종2.xlsx"
    news_wo_dup.to_excel("/media/Data/오늘의_환경뉴스/" + fn)
    
def message():
    print("스케쥴 실행중...")

news_collector()
# step3.실행 주기 설정
#schedule.every().day.at("17:00").do(news_collector)

# step4.스캐쥴 시작
#while True:
#    schedule.run_pending()
#    time.sleep(1)