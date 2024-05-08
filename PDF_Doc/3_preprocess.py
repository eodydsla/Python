import pandas as pd

# 파일 열기
input_file = 'filtered_output_excel6.xlsx'  # 입력 파일 경로
output_file = 'filtered_output_excel7.xlsx'  # 출력 파일 경로
xl = pd.ExcelFile(input_file)
writer = pd.ExcelWriter(output_file, engine='openpyxl')

# 모든 시트를 돌면서 작업 수행
for sheet_name in xl.sheet_names:
    df = xl.parse(sheet_name)

    # 필요한 연도 문자열
    required_years = ['20년', '21년', '22년', '23년', '24년']
    year_col_indices = []
    first_occurrence_row = None

    # 모든 셀을 검사하여 필요한 모든 연도가 포함된 행과 열의 인덱스를 찾기
    for index, row in df.iterrows():
        found_years = [year for year in required_years if any(year in str(cell) for cell in row)]
        if len(found_years) == len(required_years):  # 모든 연도가 포함된 행 찾기
            year_col_indices = [i for i, cell in enumerate(row) if any(year in str(cell) for year in required_years)]
            first_occurrence_row = index
            break

    if not year_col_indices:
        continue  # 필요한 모든 연도를 포함하는 행이 없으면 다음 시트로 넘어갑니다.

    # 관련 열만 추출 ('20년' 앞의 2열 포함)
    min_col_index = min(year_col_indices)
    columns_to_keep = df.columns[max(min_col_index - 2, 0): max(year_col_indices) + 1]  # 가장 큰 연도 열 인덱스까지

    # 데이터 추출
    df = df.loc[first_occurrence_row:, columns_to_keep]
    df = df.dropna(axis=1,how='all')
    # 결과를 새로운 시트에 저장
    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

# 파일 저장
writer.close()
