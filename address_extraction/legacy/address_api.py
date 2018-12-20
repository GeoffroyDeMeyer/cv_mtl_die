import requests
import json

URL = "http://maps.googleapis.com/maps/api/geocode/json"

def autocorrect_address(address_string, language="th"):
    params = {
        "address":address_string,
        "sensor":"false",
        "language":language
    }

    response = requests.get(URL, params=params)
    response = json.loads(response.content.decode())

    print(response)

    if 'error_message' in response.keys():
        print("Could not use API")
        return address_string

    elif not len(response['results']):
        print("Could not use API")
        return address_string

    else:
        addr = response['results'][0]['formatted_address']
        return addr
