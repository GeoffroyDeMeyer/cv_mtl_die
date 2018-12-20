import requests
import json

API_KEY = 'AIzaSyCPPgd6cgglR0dp1LWH-l1hU8QXDTYByJ8'

#Create the useufl URL for autocompletion address
def build_url(address, type, language):
    base_url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
    input = "?input='" + str(address) + "'"
    types = "&types=" + str(type)
    lang = "&language=" + str(language)
    key = "&key=" + API_KEY
    url = base_url + input + types + lang + key
    return url



#Input : 
# String of address you want to check into google maps database
# Type such as geocode, address 
#Language : "th" , "fr"

#Output :
# List of possible address based on autocompletion of the string in input
def prediction(addr, type , language ) :
    url = build_url(str(addr), type, language)

    response = requests.get(url)

    response = json.loads(response.content)

    response = response['predictions']


    address = []
    for elmt in response : 
        address += [elmt['description']]
    
    return address

def find_address(addr, province, district, subdistrict, township, road,plot) :
    autocorrect = autocorrect_address(addr)
    if addr == autocorrect :
        return addr
    elif province == 'กรุงเทพมหานคร' :
        return find_address_bangkok( district, subdistrict,road,plot, autocorrect)

def find_address_bangkok(district, subdistrict, road, plot, autocorrect):
    province = 'กรุงเทพมหานคร'
    if province in autocorrect and district in autocorrect and subdistrict in autocorrect and road in autocorrect  and plot in autocorrect:
        return autocorrect

    elif province in autocorrect and district in autocorrect and subdistrict in autocorrect and road in autocorrect  :
        if plot != '':
            k = 0
            search_plot = plot[k]                
            correct_address = province + ' ,' + district + ' ,' + subdistrict + ' ,' + road + ' ,' + search_plot
            while k < len(plot) and len(prediction(correct_address, type = 'address', language = 'th' )) != 0 :
                k += 1
                correct_address += plot[k]
            if k ==  len(plot) :
                correct_address = prediction(correct_address, type = 'address', language = 'th' )[0]
                return correct_address
            else :
                correct_address = correct_address[:-1]
                correct_address = prediction(correct_address, type = 'address', language = 'th' )[0]
                return correct_address
        
        else :
            return autocorrect_address(road + ' ,' + subdistrict + ' ,' + district + ' ,' + province)
        
    elif province in autocorrect and district in autocorrect and subdistrict in autocorrect :
        if plot != '':
            k = 0
            search_road= road[k]                
            correct_address = province + ' ,' + district + ' ,' + subdistrict + ' ,' + search_road
            while k < len(road) and len(prediction(correct_address, type = 'address', language = 'th' )) != 0 :
                k += 1
                correct_address += road[k]
            if k ==  len(road) :
                correct_address = prediction(correct_address, type = 'address', language = 'th' )[0]
                return correct_address
            else :
                correct_address = correct_address[:-1]
                correct_address = prediction(correct_address, type = 'address', language = 'th' )[0]
                return correct_address





