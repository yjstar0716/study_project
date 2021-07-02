import io
import pandas as pd
import seaborn as sns
from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse, HttpResponseRedirect
from .models import Ride, Weather, Weeks
from matplotlib.backends.backend_agg import FigureCanvasAgg
from networkx.drawing.tests.test_pylab import plt
from matplotlib import font_manager, rc
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

import os
import joblib
from tensorflow.keras.models import Sequential
from sklearn import preprocessing
from keras.models import load_model
from tensorflow.keras import layers, models, preprocessing
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM
import sklearn
from .models import Board

font_info = 'c:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=font_info).get_name()
rc('font', family=font_name)


def idea(request):
    rides = Ride.objects.all()  # rides 데이터를 데이터베이스에서 읽어옴
    weathers = Weather.objects.all()  # weathers 데이터를 데이터베이스에서 읽어옴

    area = request.GET.get('area')  # action으로 받아온 value값을 읽어서 저장
    years = request.GET.get('years')
    months = request.GET.get('months')  # action으로 받아온 value값을 읽어서 저장

    fig = plt.figure()
    ax = fig.add_subplot(122)

    rides_area = rides.values('days', 'weeks', 'gangwon', 'geonggi', 'gyeongnam', 'gyeongbook', 'gwangju', 'daegu',
                              'daejeon', 'busan', 'seoul', 'sejong', 'incheon', 'jeonnam', 'jeonbook', 'jeju',
                              'chungnam', 'chungbook')
    weathers_area = weathers.values('days', 'gangwon', 'geonggi', 'gyeongnam', 'gyeongbook', 'gwangju', 'daegu',
                                    'daejeon', 'busan', 'seoul', 'sejong', 'incheon', 'jeonnam', 'jeonbook', 'jeju',
                                    'chungnam', 'chungbook')

    df = pd.DataFrame.from_records(rides_area)  # 읽어온 데이터들을 데이터프레임 형태로 변환
    do = pd.DataFrame.from_records(weathers_area)

    sub_final_area = df[area]  # value값으로 받아온 값을 칼럼명에 출력하기 위해 새로 선언
    sub_final_weathers = do[area].str.split(" ", expand=True)

    rides1 = Ride.objects.filter(days__year=years,  # 년도와 월별로 나누어서 필터링
                                 days__month=months)
    weathers1 = Weather.objects.filter(days__year=years,  # 년도와 월별로 나누어서 필터링
                                       days__month=months)

    rides2 = rides1.values('days', 'weeks', 'gangwon', 'geonggi', 'gyeongnam', 'gyeongbook', 'gwangju', 'daegu',
                           # 월별 데이터를 적용해줘야하는데 Ride를 두번 사용할 수없으므로 데이터베이스에서 새로 읽어옴
                           'daejeon', 'busan', 'seoul', 'sejong', 'incheon', 'jeonnam', 'jeonbook', 'jeju',
                           # 대신 value값을 가져오는것을 rides1로 해서 월별 필터링이 완료된것을 바탕으로 가져옴
                           'chungnam', 'chungbook')

    weathers2 = weathers1.values('days', 'gangwon', 'geonggi', 'gyeongnam', 'gyeongbook', 'gwangju', 'daegu',
                                 # 월별 데이터를 적용해줘야하는데 Ride를 두번 사용할 수없으므로 데이터베이스에서 새로 읽어옴
                                 'daejeon', 'busan', 'seoul', 'sejong', 'incheon', 'jeonnam', 'jeonbook', 'jeju',
                                 # 대신 value값을 가져오는것을 rides1로 해서 월별 필터링이 완료된것을 바탕으로 가져옴
                                 'chungnam', 'chungbook')

    df2 = pd.DataFrame.from_records(rides2)  # 새로 읽어온 월별 데이터가 필터링된 데이터를 데이터프레임 형태로 변환
    do2 = pd.DataFrame.from_records(weathers2)

    final_area = df2[area]  # 월별 데이터가 필터링된 데이터에 지역명을 다시 필터링
    final_weathers = do2[area].value_counts()

    month2 = len(rides1)  # x축의 단위값을 지정하기 위해서 길이를 설정
    print(month2)  # 확인

    # 여기부터는 그래프 출력 부분
    color = sns.color_palette("plasma")
    ax = final_area.plot(kind='bar', title='일 별 지역 배달량 데이터', color=color, figsize=(20, 10), legend=False, fontsize=15)
    plt.xticks(np.arange(0, 31, step=1), ["{}".format(x + 1) for x in np.arange(0, 31, step=1)], rotation=0,
               color='darkslateblue')
    plt.yticks(color='mediumpurple')
    ax.set_ylabel('배달량', weight='bold', size=10, rotation=0, color='darkslateblue')
    ax.set_xlabel('일 별', weight='bold', size=12, rotation=0, color='darkslateblue')
    ax.set_title('일 별 지역 배달량 데이터', backgroundcolor='mediumpurple', fontsize=20, weight='bold', color='white',
                 style='italic', loc='center', pad=30)
    bx = fig.add_subplot(121)  # subplot이 다른 그래프보다 위에있으면 겹쳐서 출력안됨 #그래프 출력끝난뒤에 subplot 선언해야함
    wedgeprops = {'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}
    colors = ['silver', 'gold', 'whitesmoke', 'lightgray']
    bx = final_weathers.plot(kind='pie', title='날씨에 따른 지역별 데이터', counterclock=False, wedgeprops=wedgeprops,
                             figsize=(20, 10), colors=colors, autopct='%1.2f%%', legend=True, fontsize=15)
    plt.xticks(rotation=0)  # shadow=True
    bx.set_title('날씨에 따른 지역별 데이터', backgroundcolor='gray', fontsize=20, weight='bold', color='white', style='italic',
                 loc='center', pad=30)
    bx.set_xlabel('날씨', weight='bold', size=12, rotation=0, color='dimgray')
    bx.set_ylabel('비율', weight='bold', size=12, rotation=0, color='dimgray')

    buf = io.BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    fig.clear()
    response = HttpResponse(buf.getvalue(), content_type="image/png")
    response['Content-Length'] = str(len(response.content))

    return response


def idea_call(request):
    return render(request, 'home/idea.html')


def korea_call(request):
    chart_val = request.GET.get('chart_val')
    print(chart_val)

    weekday = request.GET.get('weekday')
    print(weekday)

    np.set_printoptions(precision=6, suppress=True)

    # ------------모델생성

    if os.path.exists("model_w.pkl"):
        print(" 모델 발견함 ")
    else:
        weeks11 = Weeks.objects.all()

        weeks_area = weeks11.values('gangwon', 'geonggi', 'gyeongnam', 'gyeongbook', 'gwangju', 'daegu',
                                    'daejeon', 'busan', 'seoul', 'sejong', 'incheon', 'jeonnam', 'jeonbook', 'jeju',
                                    'chungnam', 'chungbook')

        weeks_days = weeks11.values('week')

        df3 = pd.DataFrame.from_records(weeks_area)  # 새로 읽어온 월별 데이터가 필터링된 데이터를 데이터프레임 형태로 변환
        do3 = pd.DataFrame.from_records(weeks_days)

        df = pd.concat([df3, do3], axis=1)

        data = df[['week']]
        label = df[
            ['gangwon', 'geonggi', 'gyeongnam', 'gyeongbook', 'gwangju', 'daegu', 'daejeon', 'busan', 'seoul', 'sejong',
             'incheon', 'jeonnam', 'jeonbook', 'jeju', 'chungnam', 'chungbook']]
        model = LinearRegression()
        model = model.fit(data, label)

        joblib.dump(model, filename='model_w.pkl')
        print("모델 저장함")

    # ------------- 표 or 차트 생성

    if int(chart_val) == 1:

        model = joblib.load(filename="model_w.pkl")

        predict = model.predict([[int(weekday)]])

        y_predict = predict.reshape(-1)
        y_predict = np.round(y_predict, 1)
        print("예측 건수  : ", y_predict)

        r_predict = y_predict / y_predict.sum()
        r_predict = r_predict * 100
        r_predict = np.round(r_predict, 1)
        print("예측 비율  : ", r_predict)

        labels = ['강원', '경기', '경남', '경북', '광주', '대구', '대전', '부산', '서울', '세종', '인천', '전남',
                  '전북', '제주', '충남', '충북']

        day = ['', '월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']

        return render(request, 'home/analysis.html', {'pre': y_predict, "check": 1, 'label': labels, 'mean': r_predict, 'day': day[int(weekday)]})

    elif int(chart_val) == 2:

        model = joblib.load(filename="model_w.pkl")

        predict = model.predict([[int(weekday)]])
        print("예측 건수 : ", predict)

        y_predict = predict.reshape(-1)
        print(y_predict)

        labels = ['강원', '경기', '경남', '경북', '광주', '대구', '대전', '부산', '서울', '세종', '인천', '전남',
                  '전북', '제주', '충남', '충북']

        fig = plt.figure()

        plt.plot(labels, y_predict)
        plt.xticks(rotation=0)

        buf = io.BytesIO()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buf)
        fig.clear()
        response = HttpResponse(buf.getvalue(), content_type="image/png")
        response['Content-Length'] = str(len(response.content))
        return response
    else:
        print("욕 적지 마세요")


def home(request):
    return render(request, 'home/home.html')


def analysis(request):
    return render(request, 'home/analysis.html')


def rnn_analysis(request):
    return render(request, 'home/rnn_analysis.html')


def ann_analysis(request):
    return render(request, 'home/ann_analysis.html')


def please(request):
    rides = Ride.objects.all()
    return render(request, 'home/please.html', {'rides': rides})


def staff(request):
    return render(request, 'home/staff.html')


def pv(request):
    return render(request, 'home/pv.html')


def chart_num1(request):
    return render(request, 'home/chart_num1.html')


def rnn_call(request):
    # 공통( 예측에도 필요함 )
    chart_val = request.GET.get('chart_val')
    print(chart_val)

    area = request.GET.get('area')
    print(area)

    weeks_rnn = Weeks.objects.all()
    fig = plt.figure()

    rnn_area = weeks_rnn.values('gangwon', 'geonggi', 'gyeongnam', 'gyeongbook', 'gwangju', 'daegu',
                                'daejeon', 'busan', 'seoul', 'sejong', 'incheon', 'jeonnam', 'jeonbook', 'jeju',
                                'chungnam', 'chungbook')

    rnn_day = weeks_rnn.values('day1')

    areas = pd.DataFrame.from_records(rnn_area)
    days = pd.DataFrame.from_records(rnn_day)
    rnn = pd.to_datetime(days.index)
    # print(rnn)

    a = np.array(areas[area])  # before_normalization
    a = a.reshape(-1, 1)  # 2차원으로 reshape

    scaler = sklearn.preprocessing.MinMaxScaler()  # 0~1사이로 변경

    after_Normalization = scaler.fit_transform(a)  # after_Normalization
    # print(a, after_Normalization)

    after_Normalization = after_Normalization.reshape(-1, 5, 1)  # 5개씩 짤라서 모으고 # 3차원
    # print(after_Normalization.shape)  # 396개 데이터를 3로 나눠서 (79,5,1)

    X_train = after_Normalization[:, 0:2, 0]  # n뽑아낸 5행에서 앞 4개
    X_train = X_train.reshape(-1, 2, 1)  # 4개를 x
    Y_train = after_Normalization[:, 2, 0]  # 나머지 1개
    Y_train = Y_train.reshape(-1, 1)  # 4일치뒤의 하루를 y

    #모델생성
    if os.path.exists("mnist_mlp_model.h5"):
        print(" 모델 발견함 ")
    else:
        model = Sequential()
        model.add(SimpleRNN(2, activation="tanh", input_shape=(4, 1)))  # input_shape=(4개가들오고, 1개의열)
        model.add(Dense(1))  # 열개수
        model.compile(loss='mse', optimizer='sgd')
        model.summary()

        history = model.fit(X_train, Y_train, epochs=70, validation_split=0.2, verbose=1)
        model.save('mnist_mlp_model.h5')
        print(" 모델 생성함")

    if int(chart_val) == 1:

        model = load_model('mnist_mlp_model.h5')
        y_predict = model.predict(after_Normalization[:, 0:2, 0].reshape(-1, 2, 1))
        # 예측할때도 x데이터가 3차원으로 들어와야함.
        #print(y_predict, after_Normalization[0, 2, 0])

        town = ['', '강원', '경기', '경남', '경북', '광주', '대구', '대전', '부산', '서울', '세종', '인천', '전남',
                  '전북', '제주', '충남', '충북']

        reverseData = scaler.inverse_transform(after_Normalization[:, 2, 0].reshape(-1, 1))
        y_predict = scaler.inverse_transform(y_predict)

        y_predict = y_predict.reshape(-1)
        y_predict = np.round(y_predict)

        reverseData = reverseData.reshape(-1)
        reverseData = np.round(reverseData, 1)

        weekday = ['08.01~08.05','08.06~08.10','08.11~08.15','08.16~08.20','08.21~08.25','08.26~08.30','08.31~09.04','09.05~09.09','09.10~09.14','09.15~09.19',
                   '09.20~09.24','09.25~09.29','09.30~10.04','10.05~10.09','10.10~10.14','10.15~10.19','10.20~10.24','10.25~10.29','10.30~10.03','11.04~11.08',
                   '11.09~11.13','11.14~11.18','11.19~11.23','11.24~11.28','11.29~12.03','12.04~12.08','12.09~12.13','12.14~12.18','12.19~12.23','12.24~12.28',
                   '12.29~01.02','01.03~01.07','01.08~01.12','01.13~01.17','01.18~01.22','01.23~01.27','01.28~02.01','02.02~02.06','02.07~02.11','02.12~02.16',
                   '02.17~02.21','02.22~02.26','02.27~03.02','03.03~03.07','03.08~03.12','03.13~03.17','03.18~03.22','03.23~03.27','03.28~04.01','04.02~04.06',
                   '04.07~04.11','04.12~04.16','04.17~04.21','04.22~04.26','04.27~05.01','05.02~05.06','05.07~05.11','05.12~05.16','05.17~05.21','05.22~05.26',
                   '05.27~05.31','06.01~06.05','06.06~06.10','06.11~06.15','06.16~06.20','06.21~06.25','06.26~06.30','07.01~07.05','07.06~07.10','07.11~07.15',
                   '07.16~07.20','07.21~07.25','07.26~07.30','07.31~08.04','08.05~08.09','08.10~08.14','08.15~08.19','08.20~08.24','08.25~08.29']

        print(len(reverseData))

        return render(request, 'home/rnn_analysis.html',
                      {'pre': y_predict, "check": 1, 'mean': reverseData, "area":area, "weekday":weekday,})

    elif int(chart_val) == 2:

        model = load_model('mnist_mlp_model.h5')
        y_predict = model.predict(after_Normalization[:, 0:2, 0].reshape(-1, 2, 1))
        # 예측할때도 x데이터가 3차원으로 들어와야함.
        #print(y_predict, after_Normalization[0, 2, 0])

        reverseData = scaler.inverse_transform(after_Normalization[:, 2, 0].reshape(-1, 1))
        y_predict = scaler.inverse_transform(y_predict)

        predict = y_predict.reshape(-1)
        data = reverseData.reshape(-1)

        a = list(range(1, 80))
        np.array(a)
        plt.plot(a, predict)
        plt.bar(a, data, color='gray')

        buf = io.BytesIO()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buf)
        fig.clear()
        response = HttpResponse(buf.getvalue(), content_type="image/png")
        response['Content-Length'] = str(len(response.content))
        return response
    else:
        print("차트형태 지정에러")


def ann_call(request):
    chart_val = request.GET.get('chart_val')

    weeks_ann = Weeks.objects.all()

    area = request.GET.get('area')
    weeks_area = weeks_ann.values('gangwon', 'geonggi', 'gyeongnam', 'gyeongbook', 'gwangju', 'daegu',
                                  'daejeon', 'busan', 'seoul', 'sejong', 'incheon', 'jeonnam', 'jeonbook', 'jeju',
                                  'chungnam', 'chungbook')
    weeks_days = weeks_ann.values('week')

    areas = pd.DataFrame.from_records(weeks_area)
    days = pd.DataFrame.from_records(weeks_days)

    X_train = np.array(days)
    X_train = X_train.reshape(-1, 1)  # 학습시킬때 필요하므로 2차원으로 reshape .
    Y_train = np.array(areas[area])
    Y_train = Y_train.reshape(-1, 1)  # 학습시킬때 필요하므로 2차원 으로..

    scaler1 = sklearn.preprocessing.MinMaxScaler()  # 0~1사이로 데이터 바꿈, 데이터정규화
    X_train = scaler1.fit_transform(X_train)
    scaler2 = sklearn.preprocessing.MinMaxScaler()  # 0~1사이로 데이터 바꿈
    Y_train = scaler2.fit_transform(Y_train)

    Nin = 1  # 입력개수
    Nh = 7  # 출력 개수
    Nout = 1  # 최종출력개수

    (X_train, Y_train, scaler1, scaler2)  # 데이터를 함수에 불러오고
    transfrom = scaler1.transform([[1], [2], [3], [4], [5], [6], [7]])  # 0~1사이로 정규화

    if os.path.exists("mnist_mlp_model2.h5"):
        print(" 모델 발견함 ")
    else:
        model = models.Sequential()
        """ Keras 모델 시작 """
        model.add(layers.Dense(Nout, input_shape=(Nin,)))
        """입력 계층 노드 수 Nin 개,  은닉 계층의 노드 수 Nh 개, 활성함수는 relu  """
        model.compile(loss='mse', optimizer='sgd')
        """ cost함수 - mse – 평균 제곱 오차  최적화  알고리즘 -SGD(확률적 경사하강법) """

        history = model.fit(X_train, Y_train, epochs=50, batch_size=5,
                            validation_split=0.2, verbose=1)  # 학습용 데이터로 학습
        model.save('mnist_mlp_model2.h5')
        print(" 모델 생성함")

    model = load_model('mnist_mlp_model2.h5')
    h = model.predict([transfrom])  # 궁금한 x 데이터를 집어넣음
    # print("일주일 예측 배달건수 ", scaler2.inverse_transform(h))
    transfromer = scaler2.inverse_transform(h).reshape(-1)
    # 0~1 사이로 정규화 되었기때문에 원상복귀시킴

    labels = ['월', '화', '수', '목', '금', '토', '일']

    if int(chart_val) == 1:
        transfromer = np.round(transfromer)

        return render(request, 'home/ann_analysis.html',
                      {'pre': transfromer, "check": 1,  "area":area, "labels":labels,})

    if int(chart_val) == 2:
        fig = plt.figure()
        plt.plot(labels, scaler2.inverse_transform(h))
        buf = io.BytesIO()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(buf)
        fig.clear()
        response = HttpResponse(buf.getvalue(), content_type="image/png")
        response['Content-Length'] = str(len(response.content))
        return response

    else:
        print("r")


def blog(request, curr_page=1):
    cntPerPage = 9
    endCnt = curr_page * cntPerPage
    startCnt = endCnt - cntPerPage

    totalCnt = Board.objects.count()
    tpn = int(totalCnt/cntPerPage+1)
    totalPageCnt = []
    for i in range(1,tpn+1):
        totalPageCnt.append(i)
    board_list = Board.objects.all().order_by('-num')[startCnt:endCnt]
    context = {"board_list": board_list, "totalPageCnt": totalPageCnt, "curr_page":curr_page}
    return render(request, 'home/blog.html',context)


def detail(request,num1):
    board = get_object_or_404(Board, pk=num1)  # 하나만 리턴
    return render(request, 'home/detail.html', {'board': board})