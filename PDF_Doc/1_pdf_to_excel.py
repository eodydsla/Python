import pdfplumber
import pandas as pd

# PDF 파일 경로
file_path = '계획.pdf'

# Excel 파일로 저장할 때 사용할 writer 객체 생성
writer = pd.ExcelWriter('output_excel.xlsx', engine='openpyxl')

with pdfplumber.open(file_path) as pdf:
    # PDF의 각 페이지를 순회
    for i, page in enumerate(pdf.pages):
        # 페이지에서 테이블 추출
        # extract_tables 메소드로 테이블 데이터를 추출
        tables = page.extract_tables()
        
        for j, table in enumerate(tables):
            # DataFrame으로 변환
            df = pd.DataFrame(table[1:], columns=table[0])
            # Excel 파일에 해당 테이블 저장, 각 테이블마다 다른 시트에 저장
            df.to_excel(writer, sheet_name=f'Sheet_{i+1}_{j+1}')

# 모든 변경사항 저장 및 파일 닫기
writer.close()
