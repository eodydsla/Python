from bs4 import BeautifulSoup
import pandas as pd
import re
import requests
from tqdm import tqdm
from openai import OpenAI
import os 
from collections import OrderedDict
import json
import streamlit as st
import io
import xlsxwriter
import chardet

os.environ["OPENAI_API_KEY"] = 
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key = api_key)
st.set_page_config(layout="wide")


def remove_duplicate_words(text):
    words = text.split()
    seen = OrderedDict()
    for word in words:
        if word not in seen:
            seen[word] = None
    return ' '.join(seen.keys())

def truncate_string(text, max_length=10000):
    return text[:max_length]

    return text[:max_length]

def crawling(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    if '.pdf' in url:
        return "error"
    try:
        response = requests.get(url, headers=headers)
        print(response)
        response.encoding = 'utf-8'
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            meta = soup.find('meta', attrs={'charset': True})
            if meta != None and meta['charset'] != 'utf-8':
                response.encoding = meta['charset']
                soup = BeautifulSoup(response.content, 'html.parser')
            content = soup.get_text(strip=True)
            content = remove_duplicate_words(content)
            content = truncate_string(content)
            return content
        else:
            return "error"
    except Exception as e:
        print(e)
        return "error"

def GPTclass(x, y):
    if "확인필요" in x:
        return "O"
    y = crawling(y)
    
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = [
            {"role": "system", "content":"[[웹자료]]에 포함된 내용이 주어진 [[정보]] 관련내용이면 X, 관련내용이 아니거나, 빈페이지 또는 없는 페이지면 O 출력"},
            {"role": "user",  "content": f"[[정보]]: {x}, [[웹자료]] : {y}"}
        ]
    )
    return response.choices[0].message.content

def separator(entry):
    parts = [""] * 4
    if 'http' in entry:
        pattern_http = r',\s+(?=http)'
    else:
        pattern_http = r',\s+(?=검색일)'

    parts_http = re.split(pattern_http, entry)
    doc_info = parts_http[0]
    ref_info = parts_http[1] if len(parts_http) > 1 else ""

    pattern_doc = r'[,.] \s*(?=(?:[^"]*"[^"]*")*[^"]*$)(?=(?:[^\(]*\([^\)]*\))*[^\)]*$)(?=(?:[^“]*“[^”]*”)*[^”]*$)' 
    parts_doc = re.split(pattern_doc, doc_info)
    if len(parts_doc) == 2:
        parts[0] = parts_doc[0]
        parts[1] = parts_doc[1]
    else:
        parts[0] = parts_doc[0]

    if 'http' in ref_info:
        pattern_ref = r',\s+(?=검색일)'
        parts_ref = re.split(pattern_ref, ref_info)
        parts[2] = parts_ref[0]
        parts[3] = parts_ref[1] if len(parts_ref) > 1 else ""
    else:
        parts[3] = ref_info

    return parts

def GPTcheck(doc):
    query = """
    [[문서]]는 "출처(필요시 날짜 포함), 제목(따옴표 필수), URL, 검색일 형태로 4가지 요소로 이루어져 있고 반드시 ,로 구분하되 따옴표안 ,는 무시함
    1. [[문서]] 내용이 [[예시]]의 형태로 정리되어 있는지 체크해서 오류가 있으면 O, 없으면 X출력(4개의 요소로 구성, 콤마, 따옴표, URL 형식 등 반드시 체크) : '오류여부' 변수에 저장
    2. 출력은 반드시 JSON 포맷으로 출력해줘, 반드시 '오류여부' 변수만 존재
    
    [[예시]]
    국가법령정보센터, “물환경보전법 시행규칙”, http://www.law.go.kr/법령/물환경보전법시 행규칙, 검색일: 2018.5.3.
    국립생태원 보도자료(2017.5.26), “국립생태원, 2017년 생태공감마당 평창에서 개최”, p.8, https://www.me.go.kr/home/web/index.do?menuId=286, 검색일: 2018.7.25.
    Dutch Ministry of Infrastructure and the Environment, http://rwsenvironment.eu/subjects/soil/publications/quality-control-and/, 검색일: 2018.5.3.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": f"{query}"},
                {"role": "user", "content": f"문서:{doc}"}
            ]
        )
        result = response.choices[0].message.content
        result_dict = json.loads(result)
        result_dict["원문"] = doc  # 문서 이름 추가
        return result_dict
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

def process_entries(entries):
    articles = []
    for entry in entries:
        note = ""
        if re.search(r'(?<!")\. (?![^"]*")', entry):
            note = "확인필요"
            entry = re.sub(r'(?<!")\. (?![^"]*")', ', ', entry)
        check = separator(entry)
        check = ["확인필요" if item == 'NA' or item == '' else item for item in check]
        source = check[0]
        title = check[1]
        url = check[2]
        search_date = check[3].replace("검색일: ", "")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                url_status = "X"
            else:
                url_status = "O"
        except requests.RequestException:
            url_status = "X"
        articles.append({
            "source": source,
            "title": title,
            "URL": url,
            "search_date": search_date,
            "URL_오류여부": url_status,
            "형식체크_오류여부": note
        })
    return pd.DataFrame(articles)

def main():
    st.title("연구보고서 온라인자료 검증도구")
    uploaded_file = st.file_uploader("온라인자료 파일 업로드", type=["txt"])
    processed_data = None
    
    if uploaded_file:
        if st.button('실행'):
            data = uploaded_file.read().decode("utf-8")
            raw_entries = data.strip().split('\n')
            entries = []
            temp_entry = []
            for line in raw_entries:
                if "검색일:" in line and temp_entry:
                    entries.append(' '.join(temp_entry))
                    temp_entry = [line]
                else:
                    temp_entry.append(line)
            if temp_entry:
                entries.append(' '.join(temp_entry))

            result_df = process_entries(entries)
            GPT_check_list = [GPTcheck(doc) for doc in tqdm(entries, total=len(entries), leave=True)]
            GPT_check_df = pd.DataFrame(GPT_check_list)
            result_df['URL_오류여부'] = result_df['URL'].apply(lambda x: 'X' if x.startswith('http') else 'O')
            result_df['형식체크_오류여부'] = result_df.apply(lambda row: 'O' if '확인필요' in row.values else 'X',axis=1)            
            result_df['GPT_형식체크_오류여부'] = GPT_check_df['오류여부']
            result_df['GPT_URL_유효정보_오류여부'] = [GPTclass(x, y) for x, y in tqdm(zip(result_df['title'] + " + " + result_df['source'], result_df['URL']), total=len(result_df), leave=False)]
            result_df['원문'] = GPT_check_df['원문']

            st.dataframe(result_df)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                result_df.to_excel(writer, index=False, sheet_name='Sheet1')
            
            output.seek(0)
            processed_data = output.read()
            st.session_state.processed_data = processed_data

    # 엑셀 파일 다운로드 버튼 생성
    if 'processed_data' in st.session_state:
        st.download_button(
            label="엑셀로 다운로드",
            data=st.session_state.processed_data,
            file_name='result.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

if __name__ == "__main__":
    main()
