import airsim
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

MODEL_PTH = "./model/model.03-0.0969555.h5"

model = load_model(MODEL_PTH)

client = airsim.CarClient(ip = "192.168.1.101", port = 41451)
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

car_controls.steering = 0.0
car_controls.throttle = 0.0
car_controls.brake = 0.0

inp = np.zeros((1, 59, 255, 3))
state_inp = np.zeros((1, 3))

seq = 0
while(True):
    car_state = client.getCarState()
    if car_state.speed < 5:
        car_controls.throttle = 1.0
    else:
        car_controls.throttle = 0.0

    img_resp = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])[0]
    img_flatten = np.fromstring(img_resp.image_data_uint8, dtype = np.uint8)
    img_resp = img_flatten.reshape(img_resp.height, img_resp.width, 3)

    plt.imshow(img_resp)
    plt.savefig("./capture/1/" + str(seq) + ".png")
    seq += 1

    img_resp = img_resp[76 : 135, :255, :3].astype(float)

    inp[0] = img_resp
    state_inp[0] = np.asarray([car_controls.steering, car_controls.throttle, car_state.speed])
    output = model.predict([inp, state_inp])
    noise = np.random.uniform(0, 1, 1)
    steering = 0.005 * float(output[0][0]) + noise
    steering = np.clip(steering, -1, 1)[0]
    car_controls.steering = round(steering, 2)

    client.setCarControls(car_controls)
