import io
import requests
import random
import json
import os
from shapely.geometry import Point, Polygon
from objectdetector import generateAverageObjectList
import contextdetector
import imageeditor

#enter your API key here 
key=""

#check if street view imagery is available in a certain radius around a coordinate
def checkAvailability (lat, long, rad=60, source = "outdoor"):
    url = "https://maps.googleapis.com/maps/api/streetview/metadata?"
    url = url + "location=" + str(lat) + "," + str(long) + "&radius=" + str(rad) + "&source=" + source + "&key=" + key
    #print("making metadata request for coordinates", lat, long)
    r = requests.get(url) #make a https request to check availibilty
    if r.json()["status"] == "OK":
        #print ("location or location within radius is available")
        coordinates = r.json()["location"] #get the coordinates
        #print (coordinates["lat"],coordinates["lng"])

        if source == "outdoor":
            return round(coordinates["lat"],7), round(coordinates["lng"],7) #round coordinates
        else:
            return float (format(round(coordinates["lat"], 7), '.9f') + "1"), float (format(round(coordinates["lng"], 7), '.9f') + "1" )

    else:
        #print("no location is available within radius",rad)
        return False

#find a random street view image
def findAvailableRandomImage():
    while True:
        lat =  round (random.uniform(-90, 90),7) #get random lattitude
        long = round (random.uniform(-180, 180),7) #get random longitude
        coordinates = checkAvailability(lat,long,10000)
        if coordinates:
            #print("found available random image",coordinates)
            return coordinates

#load country shape data
def getCountriesJSON():
    return loadJSON("ExternalData\countries.geojson")

#get country polygons from shape data
def getCountryPolygons(country):
    country_polygons = []
    polygons = countries["features"][country]["geometry"]["coordinates"] #get polygons
    if len(polygons) == 1: #if there is only one polygon for the country
        country_polygons.append(Polygon(polygons[0])) #make shapely polygon
    else: #if a country consists of several polygons
        for polys in polygons:
            if len(polys) > len(polys[0]): #if single polygon
                country_polygons.append(Polygon(polys))
            else:
                for poly in polys:
                    country_polygons.append(Polygon(poly))

    return country_polygons

#get the full name and the ISO code of a country with its number
def getCountryName(country):

     return countries["features"][country]["properties"]["ADMIN"],countries["features"][country]["properties"]["ISO_A3"]

#get the number of country with its full name or ISO code
def getCountryNumber(country):
    for number in range(len (countries["features"])):
        if getCountryName(number)[0] == country or getCountryName(number)[1] == country:
            return number

#get a list with country numbers
def getCountryNumberList(country_list=[]):

    if len(country_list) == 0: #default case
        country_list = list (range(0,len(countries["features"]))) #all country numbers
    else:
        for index,country in enumerate (country_list): #for all specified countries
            if isinstance(country,str):
                country_list[index] = getCountryNumber(country) #add the country number

    return country_list

#check if point is in country
def isPointInCountry(point,polygons):
    point_adjusted = Point(point[1],point[0]) #swap lat and long for shapely
    for polygon in polygons: #check if any polygon contains point
        contains = polygon.contains(point_adjusted)
        if contains:
            return contains

    return False

#generate a random point inside a single polygon
def getPointInsideSinglePolygon(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    while True: #generate a random point inside polygons maximum bounds
        point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(point): #if point is actually inside the polygon
            return (point.y,point.x)

#randomly choose a polygon with higher chances for those with higher areas
def chosePolygon(polygons):
    weights = []
    for poly in polygons:
        weights.append(poly.area) #set polygon area as weight
    random_poly = random.choices(polygons,weights) #draw weighted
    return random_poly[0]

#get the maximum bounds of a country
def getCountryBounds(country_polygons):

    for index,polygon in enumerate (country_polygons):
        if index == 0:
            minx, miny, maxx, maxy = polygon.bounds
        else: #get the maximum bounds across all polygons
            p_minx, p_miny, p_maxx, p_maxy = polygon.bounds
            minx = min (p_minx,minx)
            miny = min (p_miny,miny)
            maxx = max (p_maxx,maxx)
            maxy = max(p_maxy,maxy)

    return minx,miny,maxx,maxy

#get a random point inside a country
def getPointInCountry(country,rad=500,rad_increment=True,attempts=10,mode=1,mode_change=True,source="outdoor"):

    polygons = getCountryPolygons(country) #get all country polygons
    counter = 0
    original_rad = rad

    if mode == 0: #get maximum country bounds
        minx,miny,maxx,maxy = getCountryBounds(polygons,rad,rad_increment)

    while True:
        if mode == 0: #search street view locations near random point inside bounds
            point = (random.uniform(miny, maxy), random.uniform(minx, maxx))
            nearest_point = checkAvailability(point[0], point[1], rad)
            if nearest_point and isPointInCountry(nearest_point, polygons):
                return nearest_point #return if actually inside country
        if mode == 1: #get random point directly from country polygons weighted on their area
            poly = chosePolygon(polygons)
            point = getPointInsideSinglePolygon(poly)
            nearest_point = checkAvailability(point[0], point[1], rad,source)
            if nearest_point and isPointInCountry(nearest_point,polygons):
                return nearest_point
        if counter >= attempts: #if no point could be found
            if mode_change and mode == 0: #change to different mode
                mode = 1
                counter = 0
                rad = original_rad
            elif mode_change and mode == 1 and source == "outdoor": #change source
                counter = 0
                rad = original_rad
                source = "default"
            else:
                return False

        if rad_increment:
            rad = rad * 2 #increase search radius

        counter = counter + 1

#get a random street view panorama from a randomly selected country
def getRandomCountryStreetView():
    while True:
        country = int(random.uniform(0, len(getCountryNumberList()))) #get a list with all countries
        point = getPointInCountry(country,50,True,7)
        if point:
            return point

#get the country  a point is located in
def getCountryofPoint(point):

    for country in range(len(getCountryNumberList())):
        polygons = getCountryPolygons(country) #get all country polygons
        if isPointInCountry(point,polygons): #check if point is in any of the polygons
            return getCountryName(country)

    return False

#collect images for a country
def saveImagesForCountry(country,number,radius=500,random_headings=True,unique_coordinates=True,x=640,y=640,heading=0,fov=90,pitch=0):
    used_coordinates = []
    index = 0
    while index < number:
        point = getPointInCountry(country,rad=radius,rad_increment=True)
        ratio = int (360/fov)
        if point:
            if random_headings:
                heading = random.randrange(0,ratio) * fov #random heading based on fov

            image = getImageWithDict(point,x,y,heading,fov,pitch)
            images = [image]

            if (unique_coordinates and image["coordinates"] not in used_coordinates): #when coordinate is new
                print ("save new coordinate")
                saveTestImages(images,getCountryName(country)[1])
                used_coordinates.append(image.copy()["coordinates"]) #store as used coordinate
                index = index + 1
            elif (not unique_coordinates):
                print("existing coordinate")
                saveTestImages(images, getCountryName(country)[1])
                index = index + 1

#check if street view is available for a country
def isStreetViewAvailable(country,api=True,attempts=10,mode_change=False):

    if api: #check if API returns a point
        available = getPointInCountry(country, rad=500, attempts=attempts,mode_change=mode_change)
    else: #check based on an external file
        coverage = loadJSON("ExternalData\coverage.json")
        if getCountryName(country)[1] in coverage:
            available = True
        else:
            available = False

    if available:
        #print("Street View available")
        return True
    else:
        #print("Street View not available")
        return False

#use API calls to update a list with street view coverage
def updateLocalCoverage(manual_list=[],attempts=10):

    coverage = loadJSON("ExternalData\coverage.json")
    if len(manual_list) == 0:
        for country in range(len(countries["features"])):
            if isStreetViewAvailable(country=country,attempts=attempts): #when api returns a point
                coverage[getCountryName(country)[1]] = True
    else:
        for country in manual_list: #manually add countries to the list
            coverage[country] = True

    saveJSON(coverage,"ExternalData\coverage")

#update the country fact sheet with information about hemisphere, used languages and so on
def updateCountryFactSheet(country_list,hemisphere=None,language_list=None,license_colors=None,domain=None):
    for country in country_list:
        path = "ExternalData/countries/" + country
        if not os.path.exists(path):
            os.makedirs(path)
        if "factsheet.json" not in os.listdir(path):
            saveJSON({},path + "/factsheet")

        factsheet = loadJSON(path + "/factsheet.json")

        if hemisphere:
            factsheet["hemisphere"] = hemisphere
        if language_list:
            factsheet["languages"] = language_list
        if license_colors:
            factsheet["license colors"] = license_colors
        if domain:
            factsheet["domain"] = domain

        saveJSON(factsheet, path + "/factsheet")

#load all countries where external data is available
def getAllCountriesWithExternalData():
    path = "ExternalData/countries/"
    return os.listdir(path)

#load all countries located within a certain hemisphere
def getCountriesByHemishphere(hemi):
    country_list = getAllCountriesWithExternalData()
    return [country for country in country_list if loadCountryFacts(country)["hemisphere"] == hemi]

#download an image via the API
def getImage(coordinates,x=640,y=640, heading=0, fov=90, pitch=0,debug=True,source="outdoor"):
    url = "https://maps.googleapis.com/maps/api/streetview?"
    if source != "outdoor" or len (str(coordinates[0]).split(".")[1]) == 10: #when source is marked as not outdoor
        source = "default"
    url = url + "&size=" + str(x) + "x" + str(y) + "&location=" + str(coordinates[0]) + "," + str(coordinates[1]) +  "&heading=" + str(heading) + "&fov=" + str(fov) + "&pitch=" + str(pitch) + "&source=" + source + "&key=" + key
    r = requests.get(url)
    if debug:
        print (url.replace(key,"hidden"))
    return r.content

#download an image via the API and store it in a dict with additional information
def getImageWithDict(coordinates,x=640,y=640,heading=0,fov=90,pitch=0,debug=True,source="outdoor"):
    return packImageDict(getImage(coordinates,x,y,heading,fov,pitch,debug,source),coordinates, heading, fov, pitch)

#save an image on hard drive
def saveImage (image,name):
    filename = name + ".png"
    with open(filename,"wb") as f:
        f.write(image)

#save a json file on hard drive
def saveJSON (obj, name):
    filename = name + ".json"
    with open(filename,"w") as f:
        json.dump(obj,f)

#save an average histogram on hard drive
def saveAverageHistogram(histogram,country,number,fov):
    path = "ExternalData/countries/" + country
    if not os.path.exists(path):
        os.makedirs(path)
    name = path + "/averageHistogram" + str(number) + "fov" + str(fov)

    saveJSON(histogram,name)

#save an average image on hard drive
def saveAverageImage(average_image,country,number,fov):
    path = "ExternalData/countries/" + country
    if not os.path.exists(path):
        os.makedirs(path)
    name = path + "/averageImage" + str(number) + "fov" + str(fov)

    saveImage(average_image,name)

#save an average word list on hard drive
def saveAverageWordList(word_list,country,number,fov,model):
    path = "ExternalData/countries/" + country

    if not os.path.exists(path):
        os.makedirs(path)
    name = path + "/averageWordList" + str(number) +  "fov" + str(fov) + "model" + model

    saveJSON(word_list,name)

#save an average object list on hard drive
def saveAverageObjectList(object_list,country,number,fov):

    path = "ExternalData/countries/" + country

    if not os.path.exists(path):
        os.makedirs(path)
    name = path + "/averageObjectList" + str(number) +  "fov" + str(fov)

    saveJSON(object_list,name)

#save several images as panorama
def savePanorama (images,name):
    print("saving panorama")
    for image in images:
        suffix = ""
        for i in image: #get suffix based on heading, pitch and field of view
            if i != "file" and i != "coordinates":
                suffix = suffix + str(i) + str(image[i])
        saveImage(image["file"],name+suffix)

#save images for later use
def saveTestImages(images,country="",dataset=""):
    folder = ""
    if country != "":
        folder = "countries/" + country + "/"
    if dataset != "":
        folder = "datasets/" + dataset + "/"
    name = "TestImages/" + folder + "la" + str(round(images[0]["coordinates"][0],7)) + "lo" + str(round(images[0]["coordinates"][1],7))
    if not os.path.exists(name):
        os.makedirs(name)

    savePanorama(images,name+"/")

#save results of a guess
def saveResults(sun,language,color,context,license,density,country,file):

    obj = {"country":country,"sun":sun,"language":language,"color":color,"context":context,"license":license,"density":density}

    try:
        already_saved = loadJSON(file + ".json")
    except:
        already_saved = []

    already_saved.append(obj)

    saveJSON(already_saved,file)




#load an image from hard drive
def loadImage (name):
    with open(name, "rb") as f:
        image = f.read()
    return image

#load a JSON file from hard drive
def loadJSON(name):
    with open(name, "rb") as f:
        obj = json.load(f)

    return obj

#check if there is already an external data directory for a country
def checkExternalDataPath(country,create=True):
    path = "ExternalData/countries/" + country
    if os.path.exists(path):
        return True
    if create:
        os.mkdir(path)
        return True

    return False

#load an average histogram from hard drive
def loadAverageHistogram(country,number,fov):

    path = "ExternalData/countries/" + country
    name = "averageHistogram" + str(number) + "fov" + str(fov) + ".json"

    average_histogram = False

    for file in os.listdir(path):
        if file == name:
           average_histogram = loadJSON(path + "/" + name)

    #print ("loading", name)

    return average_histogram

#load an average image from hard drive
def loadAverageImage(country,number,fov):

    path = "ExternalData/countries/" + country
    name = "averageImage" + str(number) + "fov" + str(fov) + ".png"

    average_image = False

    for file in os.listdir(path):
        if file == name:
           average_image = loadImageWithDict(name=name,path=path,coordinates=(0,0))

    #print("loading", name)

    return average_image

#load an average word list from hard drive
def loadAverageWordList(country,number,fov,model):

    path = "ExternalData/countries/" + country
    name = "averageWordList" + str(number) + "fov" + str(fov) + "model" +  model + ".json"

    average_wordlist = False

    for file in os.listdir(path):
        if file == name:
           average_wordlist = loadJSON(path + "/" + name)

    return average_wordlist

#load an average object list from hard drive
def loadAverageObjectList(country,number,fov,density=False):

    path = "ExternalData/countries/" + country
    name = "averageObjectList" + str(number) + "fov" + str(fov) + ".json"

    average_objectlist = False

    for file in os.listdir(path):
        if file == name:
           average_objectlist = loadJSON(path + "/" + name)

    if average_objectlist and not density:
        del average_objectlist["density"]

    return average_objectlist

#load a country fact sheet from hard drive
def loadCountryFacts(country):

    path = "ExternalData/countries/" + country

    factsheet = loadJSON(path + "/factsheet.json")

    return factsheet

#convert an image to bytes
def ImageToBytes(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()
    return image_bytes

#save temporary image
def saveTempImage(image):

    image = ImageToBytes(image)
    path = "temporary_images"
    if not os.path.exists(path):
        os.mkdir(path)
    files = os.listdir(path)
    name = path + "/" + "temp_image" +  str(len(files))
    saveImage(image,name)

#try to get pitch, fov and heading from file name
def unpackFileName(name):
    filename = name.split(".")[0]
    try:
        filename, p = filename.split("pitch")
    except:
        p = 0
    try:
        filename, f = filename.split("fov")
    except:
        f = 90
    try:
        filename, h = filename.split("heading")
    except:
        h = 0

    return int(p),int(f),int(h)

#get coordinates from folder name
def unpackCoordinates(foldername):
    try:
        foldername, lo = foldername.split("lo")
    except:
        return None

    try:
        foldername,la = foldername.split("la")
    except:
        return None

    return (la,lo)

#load an image with additional information
def loadImageWithDict(name,path=None,coordinates=None):
    if path:
        image = loadImage(path + "/" + name)
    else:
        image = loadImage(name)

    p,f,h = unpackFileName(name)
    return packImageDict(image,coordinates,h,f,p)

#generally load images
def loadGeneralImages(path):

    folders = path.split("/")
    folder_name = folders[len(folders)-2] #get the name of the image folder
    coordinates = unpackCoordinates(folder_name)

    try:
        files = os.listdir(path)
    except:
        files=[path]
        path=None

    images = []
    for f in files: #try to load all images with additional information in a dict
        images.append(loadImageWithDict(f,path,coordinates))

    return images

#load images that were stored for later use
def loadTestImages(coordinates, headings=None,fovs=None, pitchs=None,country="",dataset=""):
    folder = ""
    if country != "":
        folder = "countries/" + country + "/"
    if dataset != "":
        folder = "datasets/" + dataset + "/"

    name = "TestImages/" + folder + "la" + str(round(coordinates[0],7)) + "lo" + str(round(coordinates[1],7))
    images = []

    for img in os.listdir(name):
        p,f,h = unpackFileName(img)
        if (headings == None or int(h) in headings) and (fovs == None or int(f) in fovs) and (pitchs == None or int(p) in pitchs):
            image = packImageDict(loadImage(name+"/"+img),coordinates,h, f, p)
            images.append(image)

    return images

#load all images for a country
def loadAllCountryImages(headings=None,fovs=None, pitchs=None,country="",max=None,unique=True):
    name = "TestImages/" + "countries/" + country
    images = []

    try:
        files = os.listdir(name)
    except:
        return []

    if max: #when only a subset should be loaded
        random.shuffle(files) #shuffle order to randomize selection

    counter = 0

    for folder in files:
        temp = folder.split("la")[1]
        lat,long = temp.split("lo")
        coordinates = (float (lat),float(long))
        new_images = loadTestImages(coordinates,headings=headings,fovs=fovs,pitchs=pitchs,country=country)

        if unique and len(new_images) > 0: #make only one image is selected per coordinate
            chosen_image = random.randrange(len(new_images))
            images.append(new_images[chosen_image])
            counter = counter + 1
        else:
            for new_image in new_images:
                images.append(new_image)
                counter = counter + 1

        if max and counter == max: #when enough images have been loaded
            return images

    return images

#get the coordinates of a test dataset
def getDataSetCoordinates(dataset):

    return os.listdir("TestImages/datasets/" + dataset)

#load images from the test dataset
def loadFromTestDataset(dataset,coordinates):

    sky = loadTestImages(coordinates=coordinates,fovs=[45],pitchs=[45,90],dataset=dataset)
    horizon_one = loadTestImages(coordinates=coordinates,fovs=[45],pitchs=[0],dataset=dataset)
    horizon_two = loadTestImages(coordinates=coordinates,fovs=[90],pitchs=[0],dataset=dataset)

    return sky, horizon_one, horizon_two


#pack a dictionary with the image file and additional information
def packImageDict(file,coordinates,heading,fov,pitch):
    image = {}
    image["file"] = file
    image["coordinates"] = coordinates
    image["heading"] = heading
    image["fov"] = fov
    image["pitch"] = pitch
    return image

#request a full panorama for a location
def getFullPanorama(coordinates,x=640,y=640,fov=90):
    images = []
    pitch = -90
    while pitch <= 90:
        images.extend(getPanoramaAtPitch(coordinates,pitch,x,y,fov))
        pitch = pitch + fov
    return images

#request a limited panorama
def getPanoramaAtPitch(coordinates,pitch,x=640,y=640,fov=90,heading_start=0,heading_end=359,debug=True):
    heading = int (heading_start)
    images = []
    while heading <= int (heading_end):
        if debug:
            print ("getting image for panorama",heading,fov,pitch)
        images.append(getImageWithDict(coordinates,x,y,heading,fov,pitch,debug))
        heading = heading + fov
    return images

#acquires images representing a sky panorama
def scanSky(coordinates,x=200,y=200,pitch=45,fov=45,debug=True):
    images = getPanoramaAtPitch(coordinates,pitch,x,y,fov,debug=debug)
    images.append(getImageWithDict(coordinates,x,y,heading=360,pitch=90,fov=fov,debug=debug))
    return images

#collect images for a number of countries
def collectImages (country_list=[],number=100,radius=500,x=640,y=640,fov=90,heading=0,pitch=0,random_headings=True,unique_coordinates=True):

    country_list = getCountryNumberList(country_list)

    for country in country_list:
        print ("collecting images for ",getCountryName(country))
        if isStreetViewAvailable(country,api=False):
            if random_headings: #select a random heading
                current_number = len(loadAllCountryImages(country=getCountryName(country)[1], fovs=[fov], pitchs=[pitch],unique=unique_coordinates))
            else:
                current_number = len(loadAllCountryImages(country=getCountryName(country)[1], fovs=[fov], pitchs=[pitch], headings=[heading]),unique = unique_coordinates)

            new_number = max(0,number - current_number) #how many images need to be collected to reach a certain number
            print ("collecting new images ",new_number)
            saveImagesForCountry(country,new_number,radius,random_headings,unique_coordinates,x,y,heading,fov,pitch) #collect the images

#collect histograms for a number of countries
def collectHistograms(country_list=[],fov=90,pitch=0,number=100,update=False,unique=True):

    country_list = getCountryNumberList(country_list)

    for country in country_list:
        country = getCountryName(country)
        country = country[1]
        images = loadAllCountryImages(country=country, fovs=[fov], pitchs=[pitch],max=number,unique=unique)
        if images and checkExternalDataPath(country) and len(images) == number:
            print("generating and saving histogram for ", country)
            if update:
                saveAverageHistogram(histogram=imageeditor.getAverageHistogram(images),country=country,number=number,fov=fov)
            elif loadAverageHistogram(country=country,number=number,fov=fov) == False:
                saveAverageHistogram(histogram=imageeditor.getAverageHistogram(images), country=country, number=number,fov=fov)
            else:
                print ("histogram already exists")

#collect average images for a number of countries
def collectAverageImages(country_list=[],fov=90,pitch=0,number=100,update=False,unique=True):

    country_list = getCountryNumberList(country_list)

    for country in country_list:
        country = getCountryName(country)
        country = country[1]
        images = loadAllCountryImages(country=country, fovs=[fov], pitchs=[pitch],max=number,unique=unique)
        if images and checkExternalDataPath(country) and len(images) == number:
            print("generating and saving average image for ", country)
            if update:
                saveAverageImage(average_image=imageeditor.mergeImagesRGBNumpy(images),country=country,number=number,fov=fov)
            elif loadAverageImage(country=country,number=number,fov=fov) == False:
                saveAverageImage(average_image=imageeditor.mergeImagesRGBNumpy(images), country=country, number=number,fov=fov)
            else:
                print ("average image already exists")

#collect word lisst for a number of countries
def collectWordLists(country_list=[],fov=90,pitch=0,number=100,model="conceptual",filter=True,update=False,unique=True):

    country_list = getCountryNumberList(country_list)

    for country in country_list:
        country = getCountryName(country)
        country = country[1]
        images = loadAllCountryImages(country=country, fovs=[fov], pitchs=[pitch],max=number,unique=unique)
        if images and checkExternalDataPath(country) and len(images) == number:
            print("generating and saving word list for ", country)
            if update:
                saveAverageWordList(word_list=contextdetector.getAverageWordList(images,model,filter),country=country,number=number,fov=fov,model=model)
            elif loadAverageWordList(country=country,number=number,fov=fov,model=model) == False:
                saveAverageWordList(word_list=contextdetector.getAverageWordList(images, model, filter),country=country, number=number, fov=fov, model=model)
            else:
                print ("wordlist already exists")

#collect object lists for a number of countries
def collectObjectLists(country_list=[],fov=90,pitch=0,number=100,update=False,unique=True):

    country_list = getCountryNumberList(country_list)

    for country in country_list:
        country = getCountryName(country)
        country = country[1]
        images = loadAllCountryImages(country=country, fovs=[fov], pitchs=[pitch],max=number,unique=True)
        if images and checkExternalDataPath(country) and len(images) == number:
            print("generating and saving object list for ", country)
            if update:
                saveAverageObjectList(object_list=generateAverageObjectList(images,densities=True),country=country,number=number,fov=fov)
            elif loadAverageObjectList(country=country,number=number,fov=fov) == False:
                saveAverageObjectList(object_list=generateAverageObjectList(images, densities=True),
                                      country=country, number=number, fov=fov)
            else:
                print ("object list already exists")


#update several external data at once for a number of countries
def updateData(images=True,histograms=True,word_lists=True,object_lists=True,country_list=[],number=100,radius=500,x=640,y=640,fov=90,heading=0,pitch=0,random_headings=True,unique_coordinates=True,update=False,model="conceptual",filter=True):

    if images:
        collectImages(country_list,number,radius,x,y,fov,heading,pitch,random_headings,unique_coordinates)

    if histograms:
        collectHistograms(country_list,fov,pitch,number,update)

    if word_lists:
        collectWordLists(country_list,fov,pitch,number,model,filter,update)

    if object_lists:
        collectObjectLists(country_list,fov,pitch,number,update)

#collect a dataset for testing without api
def collectTestDataset(dataset,countries=None,all_included=True,number=2):

    if countries == None:
        countries = getAllCountriesWithExternalData()

    if all_included: #equal distribution for all countries from the list
        number = len (countries) * number
        original_countries = countries.copy()

    for index in range (number):
        country = random.choice(countries)
        if all_included:
            countries.remove(country)
            if len(countries) == 0:
                countries = original_countries.copy()

        point = getPointInCountry(getCountryNumber(country))

        sky = scanSky(point)
        horizon_one = getPanoramaAtPitch(point,0,fov=45)
        horizon_two = getPanoramaAtPitch(point,0,fov=90)

        saveTestImages(sky,dataset=dataset)
        saveTestImages(horizon_one,dataset=dataset)
        saveTestImages(horizon_two,dataset=dataset)


#clean up coordinate folders
def cleanTestImages(countries):
    for country in countries:
        print ("cleaning test images for ", country)
        name = "TestImages/" + "countries/" + country
        coordinates = os.listdir(name)
        for coordinate in coordinates:
            coordinate = name + "/" + coordinate
            images = os.listdir(coordinate)
            if len(images) != 0: #when there is more than one image for a coordinate
                kept_image = random.randrange(len(images)) #randomly select one to keep
                for index,image in enumerate(images):
                    if index != kept_image:
                        image = coordinate + "/" + image
                        os.remove(image) #delete the remaining images
            else: #when there is no image for a coordinate
                os.rmdir(coordinate) #delete the coordinate folder

#delete images that are from wrong country
def cleanWrongCoordinates(countries,delete=False,fovs=[90]):

    total_right = 0
    total_wrong = 0
    detected = []
    for index,country in enumerate (countries):
        detected.append([0,0,country])
        print ("checking for wrong coordinates for ",country)
        images = loadAllCountryImages(country=country,unique=True,fovs=fovs)
        polygons = getCountryPolygons(getCountryNumber(country))
        for image in images:
            point = image["coordinates"]
            if isPointInCountry(point,polygons): #check if image coordinate is in country
                total_right = total_right + 1
                detected[index][0] = detected[index][0] + 1
            else:
                total_wrong = total_wrong + 1
                detected[index][1] = detected[index][1] + 1
                if delete:
                    filename = "la" + str(image["coordinates"][0]) + "lo" + str (image["coordinates"][1])
                    filename = filename + "/" + "heading" + str(image["heading"]) + "fov" + str(image["fov"]) + "pitch" + str(image["pitch"]) + ".png"
                    filename = "TestImages/" + "countries/" + country + "/" + filename
                    os.remove(filename)

        print ("right coordinates",detected[index][0])
        print("wrong coordinates",detected[index][1])

    detected.sort(key=lambda x: x[1],reverse=True)

    print (detected)

    print ("totally checked images",total_right + total_wrong)
    print("totally right", total_right)
    print("totally wrong", total_wrong)


#clean invalid images
def cleanInvalidImages(countries,delete=False):

    reference_one = imageeditor.resize_images( [loadImageWithDict("reference_one.png","")],50,50)[0]
    reference_two = imageeditor.resize_images([loadImageWithDict("reference_two.png", "")], 50, 50)[0]

    for country in countries:
        print ("checking for ",country)
        images = loadAllCountryImages(country=country, unique=False)
        for image in images: #resize images and compare with reference invalid images
            image = imageeditor.resize_images([image],50,50)[0]
            #print (image["coordinates"])
            if imageeditor.compareImagesAvergeDifferenceRGB(image,reference_one) == 0 or imageeditor.compareImagesAvergeDifferenceRGB(image,reference_two) == 0:
                filename = "la" + str(image["coordinates"][0]) + "lo" + str(image["coordinates"][1])
                filename = filename + "/" + "heading" + str(image["heading"]) + "fov" + str(image["fov"]) + "pitch" + str(image["pitch"]) + ".png"
                filename = "TestImages/" + "countries/" + country + "/" + filename
                print (filename)
                if (delete):
                    os.remove(filename)





#remove duplicate countries from results
def mergeResultList(results,mode=1):

    for index,res in enumerate (results):
        remaining_results = results.copy()
        remaining_results.remove(res)

        for country in remaining_results:
            if res[0] == country[0]: #if duplicate country
                if mode == 0: #new confidence core is based on sum with upper limit
                    results[index][1] = min (1,float(results[index][1]) + float(country[1]))
                elif mode == 1 or mode == 2: #new confidence core is based on sum with no limit
                    results[index][1] = float (results[index][1]) + float (country[1])
                elif mode == 3: #new confidence core is based average
                    results[index][1] = (float(results[index][1]) + float(country[1])) / 2

                results.remove(country)


    if mode == 1 and len(results) > 0: #calculate new confidence scores based on ratio to highest score
        results.sort(reverse=True, key=lambda x: float(x[1]))
        highest_score = results[0][1]
        results = [ [country[0],(float(country[1]) / float(highest_score))] for country in results]

    if mode == 2 and len(results) > 0: #calculate new confidence scores based on average
        results = [ [country[0], float(country[1]) / 2] for country in results]

    return results


#update fact sheet for all countries
def completeFactSheetUpdate():
    updateCountryFactSheet(["ALA"], "north", ["sv"], ["white"], "ax")
    updateCountryFactSheet(["ALB"], "north", ["sq"], ["double blue"], "al")
    updateCountryFactSheet(["AND"], "north", ["ca", "es"], ["side yellow"], "ad")
    updateCountryFactSheet(["ARE"], "north", ["ar", "en"], ["white"], "ae")  # white includes alo gray
    updateCountryFactSheet(["ARG"], "south", ["es"], ["white"], "ar")
    updateCountryFactSheet(["ASM"], "tropical circle", ["sm", "en"], ["white"], "as")
    updateCountryFactSheet(["AUS"], "south", ["en"], ["white"], "au")
    updateCountryFactSheet(["AUT"], "north", ["de"], ["blue"], "at")
    updateCountryFactSheet(["BEL"], "north", ["fr", "de", "nl"], ["blue"], "be")
    updateCountryFactSheet(["BGD"], "north", ["bn"], ["blue"], "bd")
    updateCountryFactSheet(["BGR"], "north", ["bg"], ["blue"], "bg")
    updateCountryFactSheet(["BMU"], "north", ["en"], ["white"], "bm")
    updateCountryFactSheet(["BOL"], "south", ["es"], ["white"], "bo")
    updateCountryFactSheet(["BRA"], "south", ["pt"], ["white"], "br")
    # as langdetect and easyocr dont include all languages, sometimes included languages "close" to a non included language are listed
    updateCountryFactSheet(["BTN"], "north", ["dz", "hi"], ["red"], "bt")
    updateCountryFactSheet(["BWA"], "south", ["en"], ["white"], "bw")
    updateCountryFactSheet(["CAN"], "north", ["en", "fr"], ["white"], "ca")
    updateCountryFactSheet(["CHE"], "north", ["de", "fr", "it"], ["blue"], "ch")
    updateCountryFactSheet(["CHL"], "south", ["es"], ["white"], "cl")
    updateCountryFactSheet(["COL"], "tropical circle", ["es"], ["yellow"], "co")
    updateCountryFactSheet(["CRI"], "north", ["es"], ["white"], "cr")
    updateCountryFactSheet(["CUW"], "north", ["nl", "en"], ["white"], "cw")
    updateCountryFactSheet(["CZE"], "north", ["cs"], ["blue"], "cz")
    updateCountryFactSheet(["DEU"], "north", ["de"], ["blue"], "de")
    updateCountryFactSheet(["DNK"], "north", ["da"], ["blue"], "dk")
    updateCountryFactSheet(["DOM"], "north", ["es"], ["white"], "do")
    updateCountryFactSheet(["ECU"], "tropical circle", ["es"], ["yellow", "red"], "ec")
    updateCountryFactSheet(["ESP"], "north", ["es"], ["blue"], "es")
    updateCountryFactSheet(["EST"], "north", ["et", "fi"], ["blue"], "ee")
    updateCountryFactSheet(["FIN"], "north", ["fi"], ["blue"], "fi")
    updateCountryFactSheet(["FRA"], "north", ["fr"], ["blue"], "fr")
    updateCountryFactSheet(["FRO"], "north", ["fo", "da"], ["blue"], "fo")
    updateCountryFactSheet(["GBR"], "north", ["en", "ga","cy"], ["yellow"], "uk")
    updateCountryFactSheet(["GHA"], "north", ["en"], ["white"], "gh")
    updateCountryFactSheet(["GIB"], "north", ["en", "es"], ["blue"], "gi")
    updateCountryFactSheet(["GRC"], "north", ["el", "mk"], ["blue"], "gr")  # easy ocr has no greek alphabet
    updateCountryFactSheet(["GRL"], "north", ["kl", "dk"], ["white"], "gl")
    updateCountryFactSheet(["GTM"], "north", ["es"], ["white"], "gt")
    updateCountryFactSheet(["GUM"], "north", ["en"], ["white"], "gu")
    updateCountryFactSheet(["HKG"], "north", ["en", "zh-tw","zh"], ["yellow"], "hk")
    updateCountryFactSheet(["HRV"], "north", ["hr"], ["blue"], "hr")
    updateCountryFactSheet(["HUN"], "north", ["hu"], ["blue"], "hu")
    updateCountryFactSheet(["IDN"], "tropical circle", ["id"], ["black"], "id")
    updateCountryFactSheet(["IMN"], "north", ["en", "ga"], ["yellow", "side red"], "im")
    updateCountryFactSheet(["IND"], "north", ["hi", "bn", "mr", "te", "ta", "gu", "ur", "kn"], ["white"], "in")
    updateCountryFactSheet(["IRL"], "north", ["ga", "en"], ["blue"], "ie")
    updateCountryFactSheet(["ISL"], "north", ["is", "no"], ["white"], "is")
    updateCountryFactSheet(["ISR"], "north", ["he", "ar"], ["full yellow"], "il")
    updateCountryFactSheet(["ITA"], "north", ["it", "de"], ["double blue"], "it")
    updateCountryFactSheet(["JEY"], "north", ["en"], ["full yellow"], "je")
    updateCountryFactSheet(["JOR"], "north", ["ar", "en"], ["white"], "jo")
    updateCountryFactSheet(["JPN"], "north", ["ja"], ["full yellow"], "jp")
    updateCountryFactSheet(["KEN"], "tropical circle", ["en","sw"], ["white"], "ke")
    updateCountryFactSheet(["KGZ"], "north", ["ky", "ru"], ["side red"], "kg")
    updateCountryFactSheet(["KHM"], "north", ["km", "en", "ta"], ["white"], "kh")
    updateCountryFactSheet(["KOR"], "north", ["ko"], ["white"], "kr")
    updateCountryFactSheet(["LAO"], "north", ["lo", "th"], ["full yellow"], "la")
    updateCountryFactSheet(["LBN"], "north", ["ar", "fr"], ["blue"], "lb")
    updateCountryFactSheet(["LIE"], "north", ["de"], ["black"], "li")
    updateCountryFactSheet(["LKA"], "north", ["si", "ta", "en", ], ["full yellow"], "lk")
    updateCountryFactSheet(["LSO"], "south", ["en"], ["white"], "ls")
    updateCountryFactSheet(["LTU"], "north", ["lt", ], ["blue"], "lt")
    updateCountryFactSheet(["LUX"], "north", ["lb", "fr", "de"], ["blue", "full yellow"], "lu")
    updateCountryFactSheet(["LVA"], "north", ["lv"], ["blue"], "lv")
    updateCountryFactSheet(["MAC"], "north", ["zh-tw", "zh","pt"], ["black"], "mo")
    updateCountryFactSheet(["MCO"], "north", ["fr"], ["white"], "mc")
    updateCountryFactSheet(["MDG"], "south", ["mg", "fr"], ["black"], "mg")
    updateCountryFactSheet(["MEX"], "north", ["es"], ["white"], "mx")
    updateCountryFactSheet(["MKD"], "north", ["mk"], ["blue"], "mk")
    updateCountryFactSheet(["MLT"], "north", ["mt", "en"], ["blue"], "mt")
    updateCountryFactSheet(["MNE"], "north", ["sr"], ["blue"], "me")
    updateCountryFactSheet(["MNG"], "north", ["mn", "ru"], ["white"], "mn")
    updateCountryFactSheet(["MNP"], "north", ["en"], ["white"], "mp")
    updateCountryFactSheet(["MYS"], "tropical circle", ["ms", "zh-cn", "zh","ta", "en", "id"], ["black"], "my")
    updateCountryFactSheet(["NGA"], "north", ["en"], ["white"], "ng")
    updateCountryFactSheet(["NLD"], "north", ["nl"], ["full yellow"], "nl")
    updateCountryFactSheet(["NOR"], "north", ["no"], ["blue"], "no")
    updateCountryFactSheet(["NZL"], "south", ["en", "mi"], ["blue"], "nz")
    updateCountryFactSheet(["PAN"], "north", ["es"], ["white"], "pa")
    updateCountryFactSheet(["PCN"], "north", ["en"], ["white"], "pn")  # no official license plate
    updateCountryFactSheet(["PER"], "south", ["es"], ["white"], "pe")
    updateCountryFactSheet(["PHL"], "north", ["tl", "en", "es"], ["white"], "ph")
    updateCountryFactSheet(["POL"], "north", ["pl"], ["blue"], "pl")
    updateCountryFactSheet(["PRI"], "north", ["es", "en"], ["white"], "pr")
    updateCountryFactSheet(["PRT"], "north", ["pt"], ["blue", "side yellow"], "pt")
    updateCountryFactSheet(["PSE"], "north", ["ar", "en"], ["white"], "ps")
    updateCountryFactSheet(["ROU"], "north", ["ro"], ["blue"], "ro")
    updateCountryFactSheet(["RUS"], "north", ["ru"], ["white"], "ru")
    updateCountryFactSheet(["SEN"], "north", ["fr"], ["white"], "sn")
    updateCountryFactSheet(["SGP"], "tropical circle", ["en", "zh-cn", "zh", "ms", "ta", "id"], ["white"], "sg")
    updateCountryFactSheet(["SMR"], "north", ["it"], ["white"], "sm")
    updateCountryFactSheet(["SPM"], "north", ["fr"], ["white"], "pm")
    updateCountryFactSheet(["SRB"], "north", ["sr"], ["blue"], "rs")
    updateCountryFactSheet(["SVK"], "north", ["sk"], ["blue"], "sk")
    updateCountryFactSheet(["SVN"], "north", ["sl"], ["blue"], "sn")
    updateCountryFactSheet(["SWE"], "north", ["sv"], ["blue"], "se")
    updateCountryFactSheet(["SWZ"], "south", ["en"], ["white"], "sz")
    updateCountryFactSheet(["THA"], "north", ["th"], ["white"], "th")
    updateCountryFactSheet(["TUN"], "north", ["ar", "fr"], ["black"], "tn")
    updateCountryFactSheet(["TUR"], "north", ["tr"], ["blue"], "tr")
    updateCountryFactSheet(["TWN"], "north", ["zh-tw","zh"], ["white"], "tw")
    updateCountryFactSheet(["TZA"], "tropical circle", ["sw","en"], ["yellow"], "tz")
    updateCountryFactSheet(["UGA"], "tropical circle", ["en"], ["yellow"], "ug")
    updateCountryFactSheet(["UKR"], "north", ["uk", "ru"], ["white"], "ua")
    updateCountryFactSheet(["URY"], "south", ["es"], ["white"], "uy")
    updateCountryFactSheet(["USA"], "north", ["en"], ["white"], "us")
    updateCountryFactSheet(["VIR"], "north", ["en"], ["blue"], "vi")
    updateCountryFactSheet(["VNM"], "north", ["vi"], ["white"], "vn")
    updateCountryFactSheet(["ZAF"], "south", ["en", "af"], ["white"], "za")
    updateCountryFactSheet(["ZWE"], "south", ["sn","nd","en","sw"], ["yellow"], "zw")


countries = getCountriesJSON() #load country shapes




