import requests
import json


def send_positions(board):
    url = "https://tcc-xadrez.firebaseio.com/board.json"

    r = requests.put(url, json=board)
    # print("Resposta:", r.status_code, r.content)
