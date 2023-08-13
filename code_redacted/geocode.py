import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
from googlemaps import Client as GoogleMaps
from googlemaps.exceptions import ApiError, HTTPError, TransportError, Timeout

##read the google apikey string from the .env
load_dotenv()

geo_coding_key = os.getenv('geo-coding-api')
gmaps = GoogleMaps(geo_coding_key)


# create a full address from all address columns
def concat_full_address(data, address_col):
    """_summary_ Creates a column named as param address_col that is concatenated from address columns.

    Parameters
    ----------
    data : _type_ pandas.core.frame.DataFrame
        _description_ A dataframe containing columns ['stasse', 'plz', 'ort', 'bundesland']
    address_col : _type_ str
        _description_ the name of the column containing addresses
    """
    data[address_col] = data['stasse'] + ', ' + \
        data['plz'].astype('str')+' ' + data['ort'] + \
        ', ' + data['bundesland']

def fetch_geodata(data, address_col, debug=False):
    """_summary_
        Fetch geocodes (latitude, longitude) from googlemaps geocode api.
        New columns ['lat'] and ['long'] are created in your dataframe to store geocodes.
        Addresses that returned no result from googlemaps will have nan values.
        This function was tested with ~100k addresses and it works well (duration ~130 min).
        It does not fetch in windows nor does is delay requests.
        If you have a lot more data you might need to optimize.
    Parameters
    ----------
    data : _type_ pandas.core.frame.DataFrame
        _description_ the dataframe containing addresses
    address_col : _type_ str
        _description_ the name of the column containing addresses
    debug : bool, optional
        _description_, by default False. set True for seeing status output while fetching geocodes.
    """
    l = len(data)
    data['lat'] = np.NaN
    data['long'] = np.NaN
    for x in data.index:
        try:
            address = data[address_col][x]
            if(debug):
                print(f"requesting geocode for row {x}/{l} : {address}")
            # if you run into api throtteling or similar problems you can add delay.
            #time.sleep(1)
            geocode_result = gmaps.geocode(address)
            lat = geocode_result[0]['geometry']['location'] ['lat']
            long = geocode_result[0]['geometry']['location']['lng']
            if(debug):
                print(f"got result for row {x}/{l} lat,long: {lat},{long}")
            data['lat'][x] = lat
            data['long'][x] = long
        except ApiError as a:
            print("ApiError occurred.", a )
        except HTTPError as a:
            print("ApiError occurred.", a ) 
        except TransportError as a:
            print("ApiError occurred.", a ) 
        except Timeout as a:
            print("ApiError occurred.", a ) 
        except Exception as e:
            print("Unexpected error occurred.", e )