from findARestaurant import findARestaurant
from models import Base, Restaurant
from flask import Flask, jsonify, request
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, scoped_session
from sqlalchemy import create_engine

import sys
import codecs
# sys.stdout = codecs.getwriter('utf8')(sys.stdout)
# sys.stderr = codecs.getwriter('utf8')(sys.stderr)


from findARestaurant import findARestaurant
#foursquare_client_id = ''
#foursquare_client_secret = ''
#google_api_key = ''

engine = create_engine('sqlite:///restaurants.db')

Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = scoped_session(DBSession)
app = Flask(__name__)

@app.route('/restaurants', methods = ['GET', 'POST'])
def all_restaurants_handler():
  #YOUR CODE HERE
  
  if request.method == 'GET':
    return getAllRestaurants()
   
  elif request.method == 'POST':
    location = request.args.get('location', '')
    mealType = request.args.get('mealType', '')
    restaurantInfo = findARestaurant(mealType, location)

    if restaurantInfo:
      name=restaurantInfo['name']
      address=restaurantInfo['address']
      image=restaurantInfo['image']
      return makeANewRestaurant(name,address,image)
  
@app.route('/restaurants/<int:id>', methods = ['GET','PUT', 'DELETE'])
def restaurant_handler(id):
  #YOUR CODE HERE
  
  if request.method == 'GET':
    return getRestaurant(id)
    
  elif request.method == 'PUT':
    name = request.args.get('name', '')
    address = request.args.get('address', '')
    image = request.args.get('image', '')
    
    #print(id, name, address, image)
    return updateRestaurant(id, name, address, image)
    
  elif request.method == 'DELETE':
    return deleteRestaurant(id)

def getAllRestaurants():
  restaurants = session.query(Restaurant).all()
  return jsonify(Restaurant=[i.serialize for i in restaurants])

def getRestaurant(id):
  restaurants = session.query(Restaurant).filter_by(id = id).one()
  return jsonify(Restaurant=restaurants.serialize) 
  
def makeANewRestaurant(name,address, image):
  restaurant = Restaurant(restaurant_name = name, restaurant_address = address, restaurant_image = image)
  session.add(restaurant)
  session.commit()
  return jsonify(Restaurant=restaurant.serialize)

def updateRestaurant(id, name, address, image):
  restaurant = session.query(Restaurant).filter_by(id = id).one()
  
  if name is not None:
    restaurant.restaurant_name = name
  if address is not None:
    restaurant.restaurant_address = address
  if image is not None:
    restaurant.restaurant_image = image
    
  session.add(restaurant)
  session.commit()
  return "Updated a Restaurant with id %s" % id

def deleteRestaurant(id):
  restaurant = session.query(Restaurant).filter_by(id = id).one()
  session.delete(restaurant)
  session.commit()
  return "Removed restaurant with id %s" % id


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)

