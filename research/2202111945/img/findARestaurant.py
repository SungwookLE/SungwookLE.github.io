from geocode import getGeocodeLocation
import json
import httplib2

from credential import my_foursquare_api

import sys
import codecs

# sys.stdout = codecs.getwriter('utf8')(sys.stdout)
# sys.stderr = codecs.getwriter('utf8')(sys.stderr)

headers = {
    "Accept": "application/json",
    "Authorization": my_foursquare_api
}

def findARestaurant(mealType,location):
	#1. Use getGeocodeLocation to get the latitude and longitude coordinates of the location string.
	latitude, longitude = getGeocodeLocation(location)
	#print(latitude)

	#2.  Use foursquare API to find a nearby restaurant with the latitude, longitude, and mealType strings.
	#HINT: format for url will be something like https://api.foursquare.com/v2/venues/search?client_id=CLIENT_ID&client_secret=CLIENT_SECRET&v=20130815&ll=40.7,-74&query=sushi
	url = ("https://api.foursquare.com/v3/places/search?query=%s&ll=%s,%s&radius=700"\
     % (mealType,latitude,longitude))
	h = httplib2.Http()
	result = json.loads(h.request(url,'GET', headers=headers)[1])
	#print(result)
	
	if result['results']:
		#3.  Grab the first restaurant
		restaurant = result['results'][0]
		venue_id = restaurant['fsq_id'] 
		restaurant_name = restaurant['categories'][0]['name']
		restaurant_address = restaurant['location']['formatted_address']
		address = ""
		for i in restaurant_address:
			address += i + " "
		restaurant_address = address
		#print(venue_id, restaurant_name, restaurant_address)
  
		#4.  Get a  300x300 picture of the restaurant using the venue_id (you can change this by altering the 300x300 value in the URL or replacing it with 'orginal' to get the original picture
		url = ("https://api.foursquare.com/v3/places/%s/photos") % venue_id
		result = json.loads(h.request(url, 'GET', headers=headers)[1])
		#print(result)

		#5.  Grab the first image
		if result:
			firstpic = result[0]
			prefix = firstpic['prefix']
			suffix = firstpic['suffix']
			imageURL = prefix + "300x300" + suffix
		else:
			#6.  if no image available, insert default image url
			imageURL = "http://pixabay.com/get/8926af5eb597ca51ca4c/1433440765/cheeseburger-34314_1280.png?direct"
	
 		#7.  return a dictionary containing the restaurant name, address, and image url
		restaurantInfo = {'name':restaurant_name, 'address':restaurant_address, 'image':imageURL}
		print("Restaurant Name: %s" % (restaurantInfo['name']))
		print("Restaurant Address: %s" % restaurantInfo['address'])
		print("Image: %s \n" % restaurantInfo['image'])
		return restaurantInfo
	else:
		print("No Restaurants Found for %s" % location)
		return "No Restaurants Found"

if __name__ == '__main__':
	findARestaurant("Pizza", "Tokyo, Japan")
	findARestaurant("Tacos", "Jakarta, Indonesia")
	findARestaurant("Tapas", "Maputo, Mozambique")
	findARestaurant("Falafel", "Cairo, Egypt")
	findARestaurant("Spaghetti", "New Delhi, India")
	findARestaurant("Cappuccino", "Geneva, Switzerland")
	findARestaurant("Sushi", "Los Angeles, California")
	findARestaurant("Steak", "La Paz, Bolivia")
	findARestaurant("Gyros", "Sydney, Australia")