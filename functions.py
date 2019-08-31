import requests


def send_positions(x_curr, y_curr, x_next, y_nex):
    url = "https://tcc-xadrez.firebaseio.com/positions.json"

    data = {
        "current_position": {
            "x": "C",
            "y": "1"
        },
        "next_position": {
            "x": "C",
            "y": "2"
        }
    }

    r = requests.put(url, json=data)
    # print("Resposta:", r.status_code, r.content)
