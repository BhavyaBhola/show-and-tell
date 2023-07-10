import requests

url = 'http://127.0.0.1:8000/api/gen/'
file_path = r"C:\Users\91976\Desktop\programming\AI and Ml\projects\image_captioning\1007320043_627395c3d8.jpg"

with open(file_path, 'rb') as file:
    files = {'file': file}
    response = requests.post(url, files=files)

print(response.status_code)
print(response.text)