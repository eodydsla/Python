import pandas as pd
import re

# 엑셀 파일 로드
xls = pd.ExcelFile('정리2.xlsx')
final_df = pd.DataFrame()

for sheet_name in xls.sheet_names:
    print(sheet_name)
    # 각 워크시트 로드
    df = pd.read_excel(xls, sheet_name=sheet_name)
    df = df.rename(columns={df.columns[0]: '부처', df.columns[1]: '예산'})
    df['부처'] = df['부처'].fillna('환경부')
    df = df.fillna(0)
    
    # 데이터 클리닝
    df = df.applymap(lambda x: re.sub(r'\([^\)]*\)', '', x) if isinstance(x, str) else x)
    df = df.replace('\*', '', regex=True)
    df = df.replace('\*', '', regex=True)
    df = df.replace('-', 0, regex=True)
    df = df.replace(',', '', regex=True)
    df = df.replace('\(', '', regex=True)
    df = df.replace('\)', '', regex=True)
    df = df.replace('\n', '', regex=True)

        
    # 데이터 타입 변경
    df['‘20년'] = df['‘20년'].astype(int)
    df['‘21년'] = df['‘21년'].astype(int)
    df['‘22년'] = df['‘22년'].astype(int)
    df['‘23년'] = df['‘23년'].astype(int)
    df['‘24년'] = df['‘24년'].astype(int)
    
  
    # 피벗 테이블 생성
    wide_df = df.pivot_table(index='부처', columns='예산', aggfunc='sum', fill_value=0)
    
    # 멀티레벨 열 인덱스 처리
    wide_df.columns = ['{}_{}'.format(var if var else '', date if date else '') for var, date in wide_df.columns]
    wide_df['과제'] = sheet_name
    cols = wide_df.columns.tolist()  # 열 이름을 리스트로 변환
    cols = [cols[-1]] + cols[:-1]  # 마지막 열을 리스트의 첫 번째 위치로 이동
    wide_df = wide_df[cols] 
    # 최종 데이터프레임에 추가
    final_df = pd.concat([final_df, wide_df], axis=0)

# 인덱스 리셋 (부처를 열로 이동)
final_df.reset_index(inplace=True)

# 결과 저장
final_df.to_excel('processed_data.xlsx', index=False)
