def data_preprocessing2(files):
  O3_data = pd.read_csv(files[0], header=None)
  O3_data = O3_data.values
  O3_data = O3_data.reshape(8760,100,100)

  CO_data = pd.read_csv(files[1], header=None)
  CO_data = CO_data.values
  CO_data = CO_data.reshape(8760,100,100)

  NO2_data = pd.read_csv(files[2], header=None)
  NO2_data = NO2_data.values
  NO2_data = NO2_data.reshape(8760,100,100)

  QiWen_data = pd.read_csv(files[3], header=None)
  QiWen_data = QiWen_data.values
  QiWen_data = QiWen_data.reshape(8760,100,100)

  QiYa_data = pd.read_csv(files[4], header=None)
  QiYa_data = QiYa_data.values
  QiYa_data = QiYa_data.reshape(8760,100,100)

  SO2_data = pd.read_csv(files[5], header=None)
  SO2_data = SO2_data.values
  SO2_data = SO2_data.reshape(8760,100,100)

  ShiDu_data = pd.read_csv(files[6], header=None)
  ShiDu_data = ShiDu_data.values
  ShiDu_data = ShiDu_data.reshape(8760,100,100)

  weather_data = {
    'O3_data': O3_data,
    'CO_data': CO_data,
    'NO2_data': NO2_data,
    'QiWen_data': QiWen_data,
    'QiYa_data': QiYa_data,
    'SO2_data': SO2_data,
    'ShiDu_data': ShiDu_data
  }

  Size = 80

  O3_data = weather_data['O3_data']

  original_shape = O3_data.shape[1:]
  new_shape = (Size, Size)
  scale_factor = new_shape[0] / original_shape[0]


  for key in weather_data:
      weather_data[key] = rescale_data(weather_data[key], scale_factor, new_shape)

  O3_data = weather_data['O3_data']

  normalized_weather_data = {key: normalize_dataset(data) for key, data in weather_data.items()}
  weather_data = normalized_weather_data

  O3_data = weather_data['O3_data']


  n_steps = 7   #12 24 48...
  x_train, y_train = generate_dataset(weather_data, O3_data, n_steps)
  x_train = x_train.reshape((x_train.shape[0], n_steps, Size, Size, len(weather_data)))
  split_index = int(len(x_train) * 0.9)
  x_train, x_test = x_train[:split_index], x_train[split_index:]
  y_train, y_test = y_train[:split_index], y_train[split_index:]

  return x_train, y_train, x_test, y_test