import requests

url = "http://127.0.0.1:8000/predict"

data = {"size": 102.73, "nb_rooms": 2, "garden": 1}
response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Response JSON:", response.json())