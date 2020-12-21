import pandas as pd  # 다양한 파일 형식의 데이터 가져올 수 있게 하는 패키지
import numpy as np  # 연산을 위한 패키지
import matplotlib.pyplot as plt  # 그래프를 그리게 해주는 패키지
import math # 복잡한 연산을 위한 패키지

# 해당 링크는 한국거래소에서 상장법인목록을 엑셀로 다운로드하는 링크입니다.
# 다운로드와 동시에 Pandas에 excel 파일이 load가 되는 구조입니다.
df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0)[0]
# df.head()

# 데이터에서 정렬이 따로 필요하지는 않지만 테스트겸 Pandas sort_values를 이용하려 정렬을 시도해봅니다.
df.sort_values(['상장일'], ascending=True)

# 필요한 것은 "회사명"과 "종목코드" 이므로 필요없는 column들은 제외
df = df[['회사명', '종목코드']]

# 한글 컬럼명을 영어로 변경
df = df.rename(columns={'회사명': 'company', '종목코드': 'code'})
# stock_code.head()

# 종목코드가 6자리이기 때문에 6자리를 맞춰주기 위해 설정해줌
df.code = df.code.map('{:06d}'.format)

# 종목 입력 받기
stocklist = []
print("======================================================================================")
print("투자하고자 하는 종목을 하나씩 차례로 입력해주세요. \n(마지막 종목 입력 후, 추가로 뜨는 창에 '끝'이라고 입력해주시면 입력을 종료하고 다음 단계로 넘어갑니다.)")
print("--------------------------------------------------------------------------------------")

# 종목 입력시에 소문자는 모두 대문자로 인식하게 만들기
while True:
    stock = input("투자하실 종목을 입력하세요 : ")
    if stock == "끝":
        break
    stocklist.append(stock.upper())

print("--------------------------------------------------------------------------------------")
print("다음 단계로 넘어갑니다.")
print("--------------------------------------------------------------------------------------")

# 총 투자 금액과 주식 투자 비중 입력 받기
asset = int(input("주식과 채권에 투자하실 금액을 입력하세요. (숫자만 입력해주시면 됩니다.): "))
A = int(input("주식투자 비중을 얼마로 하시겠습니까? (0~100% 사이의 숫자만 입력해주시면 됩니다.): ")) / 100
print("--------------------------------------------------------------------------------------")

stks = int(len(stocklist))

STKSDATA = np.zeros((3, stks))  # 행을 3개로 만들어서 각각의 행에 해당 주식종목의 평균, 분산, 오늘의 가격을 입력받음
STKSRET = pd.DataFrame(index=range(0, 2000))

# 입력 받은 종목들 자료 가져오기(크롤링)
for i in range(stks):
    stock1 = stocklist[i]
    print("%s의 정보를 가져오고 있습니다. 잠시만 기다려주십시오." % stock1)

    # 입력한 회사에 대해서 page를 반복해서 대량의 데이터 가져오기
    company = stock1
    code = df[df.company == company].code.values[0].strip()

    df1 = pd.DataFrame()
    for page in range(1, 50):  # 페이지 범위 설정하기 - 충분한 데이터를 가져오기 위해 50페이지까지의 데이터를 가져옴
                               # 해당 코드를 test할 때에는 50보다 작은 수를 입력하여 코드 실행 속도를 높일 수 있음
        url = 'http://finance.naver.com/item/frgn.nhn?code={code}'.format(code=code)
        url = '{url}&page={page}'.format(url=url, page=page)
        # print(url)
        html = pd.read_html(url, header=0, encoding='euc-kr')  # 한글 깨지는 것 해결하기 위해 인코딩 'euc-kr' 설정
        df1 = df1.append(pd.read_html(url, header=0, encoding='euc-kr')[2], ignore_index=True)

    print("정보를 처리 중입니다. 잠시만 기다려주십시오.")

    # 데이터 다루기
    df1 = df1.dropna()  # 데이터 클린징 # df.dropna()를 이용해 결측값 있는 행 제거

    # 종가
    df1_price = df1.loc[:, ['종가']]  # 필요한 열만 남기기
    df1_price = df1_price.rename(columns={'종가': 'close'})  # 컬럼명을 영어로 바꿔줌

    # 등락률
    df1_sample = df1.loc[:, ['등락률']]
    df1_sample = df1_sample.rename(columns={'등락률': 'rr'})
    df1_sample['rr'] = df1_sample['rr'].str.replace('%', '')  # str열에 있는 부호(%) 없애주기

    # 0번째 인덱스에 셀의 str을 숫자 0으로 바꾼 후 없애주기  # 이렇게 하지 않으면 int로 바꾸는 과정에서 오류남
    df1_price['close'] = np.where(df1_price['close'] == '종가', 0, df1_price['close'])
    df1_price['close'] = df1_price['close'].astype(int)  # str을 int으로 바꿔주기
    df1_price = df1_price.drop(df1_price.index[0])  # 0번째 인덱스 없애기
    pp1 = df1_price.iloc[2]
    k = int(df1_price['close'][2])  # 해당 주식의 제일 최근 종가를 입력받아서 결과값으로 몇 주의 주식을 매수할 것인지에 대한 정보를 알려줌
    df1_sample['rr'] = np.where(df1_sample['rr'] == '등락률', 0, df1_sample['rr'])
    df1_sample['rr'] = df1_sample['rr'].astype(float)  # str을 float으로 바꿔주기
    df1_sample = df1_sample.drop(df1_sample.index[0])

    # weight와 efficient frontier 계산을 위한 평균, 표준편차 구하기
    df1_mean = df1_sample.mean()
    df1_var = df1_sample.var()

    STKSDATA[0][i] = df1_mean
    STKSDATA[1][i] = df1_var
    STKSDATA[2][i] = k

    STKSRET[stock1] = df1_sample

STKSRET = STKSRET.dropna(axis=0)

# 상수 값 설정
n = 20000  # 충분히 큰 난수로 우리가 임의로 설정 , 해당 코드에서는 20000개의 비중에 대한 난수를 생성함
m = 300   # 무위험채권을 고려한 포트폴리오의 기대수익률과 sigma를 나타내주는 난수의 수
rf = 0.63 / 365  # CD-91물의 제일 최근 (2020년 10월) 평균 금리가 0.63%였고 이를 하루 금리로 환산한 값
PF = np.zeros((3, n))
weight = np.zeros((n, stks))
for i in range(n):

    weights = np.random.uniform(0, 1, stks)  # 난수를 생성할 때 공매도는 불가능하다는 조건으로 난수생성
                                             # 공매도가 가능하다는 조건을 넣을때는 0,1 대신에 음수의 값 사용 가능
    if sum(weights) != 0:                    # 매우 낮은 가능성이지만 비중의 합이 0이 될 수 있으므로 그 값은 제외시켜줌

        weigh = weights / sum(weights)      # 포트폴리오의 각 비중의 합을 1이 되도록 해주는 기법 

        PF[0, i] = np.sum(weigh * STKSDATA[0][:])                 # 랜덤으로 생성된 비중 별 포트폴리오의 기대수익률
        port_var = np.dot(weigh.T, np.dot(STKSRET.cov(), weigh))  # 랜덤으로 생성된 비중 별 포트폴리오의 분산
        PF[1, i] = np.sqrt(port_var)                              # 랜덤으로 생성된 비중 별 포트폴리오의 표준편차
        PF[2, i] = (PF[0, i] - rf) / PF[1, i]                     # 랜덤으로 생성된 비중 별 포트폴리오의 Sharpe Ratio

        for j in range(stks):
            weight[i][j] = weigh[j]
    else:
        pass
plt.scatter(PF[1, :], PF[0, :], s=1, color='red')

MP = max(PF[2, :])     # Sharpe Ratio가 최대가 되는 점 - 이때의 비중이 위험대비 기대수익률이 가장 큰 포트폴리오의 최적 비중이 된다.

a = weight[np.where(PF[2, :] == MP)][:]

retP = PF[0, np.where(PF[2, :] == MP)][0][0]  # 최적 포트폴리오의 Expected return
stdP = PF[1, np.where(PF[2, :] == MP)][0][0]  # 최적 포트폴리오의 Standard Deviation
plt.scatter(PF[1, np.where(PF[2, :] == MP)], PF[0, np.where(PF[2, :] == MP)], color='red')



MYPF = np.zeros((2, m))
# m을 충분히 크게 설정해 줌으로써 risk free rate과 최적의 포트폴리오의 기대수익률을 잇는 직선을 나타냄
# 그림의 이해를 돕기 위해 np.linspace의 끝의 값을 retP-rf, stdP 로 설정하지 않고 2 * retP - rf , 2 * stdP의 값을 사용하였음

MYPF[0, :] = np.linspace(rf, 2 * retP - rf, m)  # risk free rate과 최적 포트폴리오의 Expected return을 m개의 점으로 분할하여 직선으로 표현
MYPF[1, :] = np.linspace(0, 2 * stdP, m)  # 0과 최적 포트폴리오의 sigma를 m개의 점으로 분할하여 직선처럼 표현

plt.scatter(PF[1, :], PF[0, :], s=1)

plt.scatter(MYPF[1, :], MYPF[0, :], s=1) 
plt.scatter(A*stdP, A*retP, marker="*", s=500, color='black')  # 무위험채권을 고려한 최적포트폴리오의 sigma와 기대수익률을 나타내는 점
plt.xlabel('sigma')  # x축 라벨명 지정하기
plt.ylabel('Expected rate of return')  # y축 라벨명 지정하기
plt.xlim(0, 5)  # x축 범위 설정 : 0~5로 확대
plt.ylim(0, 1)  # y축 범위 설정 : 0~1로 확대
plt.grid(True)  # x,y축에 대한 그리드 표시
plt.axis('tight')  # 모든 데이터를 볼 수 있을 만큼의 축 범위 설정
plt.title('Optimal Portfolio Graph', fontsize=18, color='blue')  # 그래프 제목, 글자크기, 파란색 지정


for i in range(stks):
    a[0][i - 1] = float(a[0][i - 1])

print("결과를 가져오고 있습니다. 잠시만 기다려주십시오.")
print("--------------------------------------------------------------------------------------")
print("채권에 {:,}원 투자하세요. ".format(int((1 - A) * asset)))

for i in range(stks):
    a[0][i - 1] = float(a[0][i - 1])
for i in range(0, stks):
    print("{}에 {:,}원 투자하세요. ".format(stocklist[i - 1], int(A * asset * a[0][i - 1])))
    print("{}의 현재 주가는 {:,}원입니다. {:d}주 매수하세요.".format(stocklist[i - 1], STKSDATA[2][i - 1],
                                                     math.floor(int(A * asset * a[0][i - 1]) / STKSDATA[2][i - 1])))
print("--------------------------------------------------------------------------------------")

plt.show()  # 그림 표시

print("이용해 주셔서 감사합니다!")
print("======================================================================================")
