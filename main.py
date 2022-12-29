import tensorflow as tf                                            #Библиотека для обучения нейронной сети
import numpy as np                                                 #Библиотека для работы с массивами
import matplotlib.pyplot as plt                                    #Библиотека визиулаизации данных
import imageio                                                     #Библиотека для работы с изображениями

mnist = tf.keras.datasets.mnist                                    #Сассивы с данными mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()           #1.Тренеровочный массив(происходит обучение) 2.Тестовый массив(проверка обучения)

x_train = x_train / 255                                            #Изменяем яркость пикселей от 0 до 1
x_test = x_test / 255                                              #Изменяем яркость пикселей от 0 до 1

plt.figure(figsize=(8,8))                                          #Здесь происходить отрисовка первых 16 цифр
for i in range(16):
  plt.subplot(4,4, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(x_train[i], cmap=plt.cm.binary)
  plt.colorbar()
  plt.xlabel(y_train[i])
plt.show()

model = tf.keras.models.Sequential([                               #Создаем модель нейронной сети
    tf.keras.layers.Conv2D(                                        #Коммуникационный слой (выделяет признаки) (сверточные нейронные сети)(Определяетя направления наших прямых)
        input_shape=(28, 28, 1),                                   #Размер изображения 28x28 пикселя
        filters=32,                                                #Кол-во фильтров
        kernel_size=(5, 5),                                        #Размер ядра 5x5 пикселей
        padding='same',                                            #Для того что-бы наш фильтр выходил за пределы изображения и ее изменял его
        activation='relu',                                         #Функция активации
    ),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),                   #Слой пулинга. С размером ядра 2x2 пикселя
    tf.keras.layers.Conv2D(                                        #Добавляем слой, для улучшений н.с. (Складиывает углы,благодаря этим углам будем более точно определять цифры)
        filters=64,                                                #Увеличиваем фильтр в 2 раза до 64
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    ),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),                  #Слой пуллинга усиливает выделенные нами признаки, для того чтоб мы  могли лучше различать цифры
    tf.keras.layers.Flatten(),                                    #Слой трансформации
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),           #Скрытый с 1024 нейронами (слой для улучшения нейронной сети)
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)           #Полносвязный слой: 1)Кол-во выходных нейронов. 2)Функция активации.
])

model.compile(                                                    #Компиляция модели с 3 параметрами.
    optimizer='adam',                                             #Оптимизатор
    loss='sparse_categorical_crossentropy',                       #Потери (считает ошибку нейронной сети)
    metrics=['accuracy']                                          #Метрика (точность)
)

model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=5)     #Обучения нейронной сети. Добавляем размерность для ком.слоя метод ришейп.
                                                                 #Имеются 3 параметра.
                                                                 #1.Тренеровочный набор данных(изображения) 2.Ответы на изображения
                                                                 #3.Кол-во раз прогоняемых тестовый набор данных

print(model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test))     #Проверка качества обучений нейронной сети на тестовом наборе данных

def model_answer(model, filename, display=True):                 #Функция с 3 параметрами (1.Наша обученая модель н.с. 2.Имя изображения. 3.Дисплей для отрисовки изображения)
  image = imageio.imread(filename)                               #Загружаем наши изображения.
  image = np.mean(image, 2, dtype=float)                         #Преобразовываем изображения.
  image = image / 255
  if display:                                                    #Отрисовка изображения с использованием библиотеки matplotlib
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(filename)
    plt.show()

  image = np.expand_dims(image, 0)
  image = np.expand_dims(image, -1)
  return np.argmax(model.predict(image))

for i in range(10):                                             #Загружаем изображения с помощью цикла
  filename = f'{i}.png'
  print('Имя файла:' , filename, '\tОтвет сети:' , model_answer(model, filename, False))
print(model_answer(model, '1.png'))