# unused
import json
import os

if os.path.isfile("./src/settings.local.json"):
    with open('./src/settings.local.json', 'r') as f:
        settings = json.load(f)
else:
    with open('./src/settings.json', 'r') as f:
        settings = json.load(f)
