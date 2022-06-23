import random
import json
import pickle
import numpy as np
import base64
import datetime
from keras.utils import load_img, img_to_array
from keras.models import load_model
from flask import Flask, render_template, request
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json', encoding='utf8').read())
words   = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model   = load_model('language_model8.h5')
foodmodel = load_model('foodmodel5.h5')
foodclasses = ['Bánh bèo','Bánh bột lọc','Bánh căn','Bánh canh','Bánh chưng',
               'Bánh cuốn','Bánh đúc','Bánh giò','Bánh khọt','Bánh mì',
               'Bánh pía','Bánh tét','Bánh tráng nướng','Bánh xèo','Bún bò Huế',
               'Bún đậu mắm tôm','Bún mắm','Bún riêu','Bún thịt nướng','Cá kho tộ',
               'Canh chua','Cao lầu','Cháo lòng','Cơm tấm','Gỏi cuốn',
               'Hủ tiếu','Mì quảng','Nem chua','Phở','Xôi xéo']

app = Flask(__name__)

@app.route("/")
def home():
  return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
  message = request.form["msg"]
  exts = ["jpg", "png", "jpeg"]
  tim = datetime.datetime.now()
  tim = tim.strftime("%y%m%d-%H-%M-%S")
  for ext in exts:
    x = "data:image/"+ext+";base64,"
    if(message.find(x) != -1):
      image_path = "./images/img_"+tim+"."+ext
      message = message.replace(x,"",1)
      img_decode = base64.b64decode(message)
      imgmsg = open(image_path,'wb')
      imgmsg.write(img_decode)
      detection = food_detect(image_path)
      message = detection
      break
  print(message)
  print("Processing...")
  ints = predict_class(message)
  res = get_response(ints, intents)
  print(res)
  return res

def food_detect(image_path):
  img = load_img(image_path, target_size=(150,150))
  img = img_to_array(img)
  img = img.reshape(1,150,150,3)
  img = img.astype('float32')
  img = img/255
  detection = foodclasses[np.argmax(foodmodel.predict(img))]
  return detection

def clean_up_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence.lower())
  sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
  return sentence_words

def bag_of_words(sentence):
  sentence_words = clean_up_sentence(sentence)
  bag = [0] * len(words)
  for w in sentence_words:
    for i, word in enumerate(words):
      if word == w:
        bag[i] = 1
  return np.array(bag)

def predict_class(sentence):
  bow = bag_of_words(sentence)
  res = model.predict(np.array([bow]))[0]
  ERR_THRESHOLD = 0.5
  results = [[i,r] for i, r in enumerate(res) if r > ERR_THRESHOLD]
  results.sort(key = lambda x: x[1], reverse=True)
  return_list = []
  for r in results:
    return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})
  if return_list == []:
    return_list.append({'intent': 'khonghieu', 'probability': str(1) }) 
  return return_list

def get_response(intents_list, intents_json):
  tag = intents_list[0]['intent']
  list_of_intents = intents_json['intents']
  for i in list_of_intents:
    if i['tag'] == tag:
      result = random.choice(i['responses'])
      break
  return result

if __name__ == "__main__":
  app.run(host="localhost", port=8050)