from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from PIL import Image
import urllib.request, time, threading, glob, tensorflow as tf, pandas as pd, matplotlib.pyplot as plt, numpy as np, os

path = "chrome_driver(230206)\\chromedriver.exe"
driver_tr1 = webdriver.Chrome(path)
driver_tr2 = webdriver.Chrome(path)
driver_tr3 = webdriver.Chrome(path)
driver_tr4 = webdriver.Chrome(path)
driver_tr5 = webdriver.Chrome(path)

def find_park(z, driver_tr):
    url = "https://www.google.co.kr/imghp?hl=ko&ogbl"
    driver_tr.get(url)
    im_search = driver_tr.find_element("name","q")
    im_search.send_keys(z)
    im_search.send_keys(Keys.RETURN)

    #스크롤 내리는 코드 추가
    SCROLL_PAUSE_TIME = 1
    last_height = driver_tr.execute_script("return document.body.scrollHeight")
    get_count = 0
    running1 = True
    while running1:
        get_count+=1
        driver_tr.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver_tr.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver_tr.find_element_by_css_selector(".mye4qd").click() #결과 더보기 버튼 
            except:
                break
        last_height = new_height
        if get_count == 3: break

    K_imgs = driver_tr.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")
    

    def createDirectory(dir):
        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
        except OSError:
            print("디렉토리 생성 실패".format(dir))

    createDirectory(z)
    
    
    count = 1
    for K_img in K_imgs:
        try:
            K_img.click()
            time.sleep(2)
            K_img_big = driver_tr.find_element(By.XPATH, "/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div[2]/div/div[2]/div[2]/div[2]/c-wiz/div/div[1]/div[2]/div[2]/div/a/img")
            K_img_big_url = K_img_big.get_attribute("src")
            
            opener = urllib.request.build_opener()
            opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(K_img_big_url, z + "/" + str(count) + ".png")
            Image.open(z + "/" + str(count) + ".png").resize((100,100), Image.ANTIALIAS).save(z + "/" + str(count) + ".png", "png")
            im_small_tr = plt.imread(z + "/" + str(count) + ".png")
            im_small_li = np.array(im_small_tr.shape)
            if im_small_li[-1] != 3:
                continue
            count += 1
            print(count)
            if count == 50:
                running1 = False
                break
            
        except: #저작권에 의한 다운 실패 시
            pass
        
    driver_tr.close()
    

tr1 = threading.Thread(target=find_park, args=("구름표범", driver_tr1))
tr1.daemon = True
tr1.start()
tr2 = threading.Thread(target=find_park, args=("오셀롯", driver_tr2))
tr2.daemon = True
tr2.start()
tr3 = threading.Thread(target=find_park, args=("치타", driver_tr3))
tr3.daemon = True
tr3.start()
tr4 = threading.Thread(target=find_park, args=("표범", driver_tr4))
tr4.daemon = True
tr4.start()
find_park("뱅갈", driver_tr5)


print("4초 뒤 ML Start!!")
time.sleep(4)   # ML 텀 적용


#(1) 데이터
paths = glob.glob('.\\*\\*.png')
paths = np.random.permutation(paths)
    
independent = np.array([plt.imread(paths[i]) for i in range(len(paths))])
dependent = np.array([paths[i].split('\\')[-2] for i in range(len(paths))])

dependent = pd.get_dummies(dependent)    # 원핫인코딩

#(2) 모델
X = tf.keras.layers.Input(shape=[100, 100, 3]) #3차원으로 변경

H = tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='swish')(X) #컨벌루션 mask1 적용
H = tf.keras.layers.MaxPool2D()(H) # 풀링1
H = tf.keras.layers.Conv2D(16, kernel_size=5, activation='swish')(H) #컨벌루션 mask2 적용
H = tf.keras.layers.MaxPool2D()(H) # 풀링2

H = tf.keras.layers.Flatten()(H) # 평탄화
H = tf.keras.layers.Dense(120, activation='swish')(H)
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(5, activation="softmax")(H)
model = tf.keras.models.Model(X,Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')


#(3) 학습 
model.fit(independent[10:], dependent[10:], epochs=50)


#(4) 검증( 예측값 : 원본값 )
print("< 판단값 >")
pre = model.predict(independent[:10])
print(pd.DataFrame(pre).round(2))

print("< 실제값 >")
print(dependent[:10])
print(paths[:10])