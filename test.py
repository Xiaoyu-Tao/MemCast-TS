import http.client
import json

conn = http.client.HTTPSConnection("api2.aigcbest.top")
payload = json.dumps({
   "model": "deepseek-ai/DeepSeek-R1",
   "messages": [
      {
         "role": "user",
         "content": "Hello!"
      }
   ]
})
headers = {
   'Accept': 'application/json',
   'Authorization': 'Bearer sk-PxM40luD13UVKLhp6k3zenHC2XPASEi5uazXuXsCfTrQ3hUQ',
   'Content-Type': 'application/json'
}
conn.request("POST", "/v1/chat/completions", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))