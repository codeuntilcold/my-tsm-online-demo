import cv2
import base64
import requests

class Connector():
    def __init__(self):
        pass

    def send_data(self, image, output, output_aod):
        _, jpeg = cv2.imencode('.jpeg', image)
        base64_jpeg = base64.b64encode(jpeg)
        res = requests.post('http://127.0.0.1:5001/send-frame',
                      json={
                          "image": base64_jpeg.decode(),
                          "data": output,
                          "object": output_aod
                      })
        return res.json()['success']