import easyocr
from langdetect import detect_langs
from lingua import Language,LanguageDetectorBuilder
from difflib import SequenceMatcher
from shapely.geometry import Point,box
from geotext import GeoText
import pycountry
import imageeditor
import datamanager


#dictionary with the alphabet number as key and all languages of the alphabet as words
config_dict = {
        0: ['en','fr','es','de','af','cs','cy','da','et','ga','hr','hu','is','it','lt','lv','mt','nl','no','pl','pt','ro','rs_latin','sk','sl','sq','sv','tr','ms','id','mi'],
        1: ['ko'],
        2: ['bg','mn','ru','rs_cyrillic','tjk','uk'],
        3: ['ch_sim'],
        4: ['ch_tra'],
        5: ['ar'],
        6: ['ja'],
        7: ['th'],
        8: ['ta'],
        9: ['bn'],

    }


Textreader = None #the currently used textreader

TextReaders = [] #global list containing all initialised textreaders

lingua_detector = None #the detector model when using lingua

#initialise as much textreaders as needed for the alphabets
def initTextReaders(alphabets=[0,1,2,3,4,5,6,7,8,9]):
    print ("loading alphabets for reader")
    for alphabet in config_dict:
        if alphabet in alphabets:
            print("loading alphabet ",alphabet)
            TextReaders.append(easyocr.Reader(config_dict.get(alphabet))) #initialise and add to the global list
        else:
            TextReaders.append(None)

#change the used text reader for the reading operations
def changeReader(alphabet,perfomance_mode=True):
    global Textreader
    if perfomance_mode:
        Textreader = TextReaders[alphabet] #use a preinitialised reader
    else:
        Textreader = easyocr.Reader(config_dict.get(alphabet)) #reinitialise reader

#read the text in an image
def readText(image):

    resultsraw = Textreader.readtext(image, detail=1)

    return resultsraw

#filter out the word "Google" (from possible watermarks), all words below a confidence threshold and all numbers
def filterResults(resultsraw,threshold=0.1,filter_numbers = True):

    results_filtered = []
    for result in resultsraw: #for all words
        filter_similarity = max(SequenceMatcher(None, result[1], "Google").ratio(),SequenceMatcher(None, result[1], "6ооде").ratio(),SequenceMatcher(None, result[1], "@০০9ট").ratio())
        if filter_similarity < 0.4 and result[2]>threshold and not (result[1].isnumeric() and filter_numbers): #if not too close to "Google", confidence high enough and not numeric
            results_filtered.append(result) #keep the word in results

    return results_filtered

#get the average confidence of a guess
def getConfidence(results):
    total_confidence = 0
    for result in results: #for all words
        total_confidence = total_confidence + result[2] #add up the confidence
    try:
        return total_confidence/len(results) #return the average
    except: #if result is empty
        return 0

#get the most likely alphabets
def getMostConfident(image,alphabets=[0,1,2,3,4,5,6,7,8,9,10],image_filter=True):
    detected_alphabet = [-1,-1,[]] #list to later include alphabet number, average confidence and text
    results = []
    if image_filter:
        image = imageeditor.blurWatermark(image)
    for alphabet in alphabets: #iterate over all used alphabets
        changeReader(alphabet)
        detected_alphabet[0] = alphabet #get alphabet number
        text = filterResults(readText(image["file"])) #read text and filter results
        detected_alphabet[1] = getConfidence(text) #get average confidence
        detected_alphabet[2] = text #get text
        results.append(detected_alphabet.copy())

    results.sort(reverse=True,key=lambda x: x[1]) #sort by confidence

    return results[:2] #return the two results with the highest confidence values

#add all words to a longer text for each alphabet
def getCompleteText(text):
    alphabets = []
    for alphabet in text: #iterate over all alphabet numbers in the dictionary
        text_complete = ""
        for word in text.get(alphabet): #iterate over each single word in text for each alphabet
            text_complete = text_complete + word + " "
        alphabets.append([text_complete,len(text.get(alphabet))]) #append list with complete text and number of words

    return alphabets



#get the most likely languages with lingua
def detectLanguageLingua(text,thresh=0.4):

    global lingua_detector

    if lingua_detector == None: #if the detector has not yet been build
        print ("building lingua detector for the first time")
        lingua_detector = LanguageDetectorBuilder.from_all_languages().build()

    languages = lingua_detector.compute_language_confidence_values(text)[0:4] #focus on languages with highest confidence
    lowest_confidence = min (0.99,languages[len(languages) - 1][1])

    results = []

    for language in languages: #change to langdetect format and spread out confidences
        new_confidence = (language[1] - lowest_confidence) / (1 - lowest_confidence)
        if new_confidence > thresh: #filter out all languages with a low confidence
            results.append(language[0].iso_code_639_1.name.lower() + ":" + str(new_confidence))


    return results


def combineLanguageResults(lang,lingua,mode=2):

    combined_list = []

    for langauge in lang[0]: #convert to list
        language_code,confidence = str(langauge).split(":")
        combined_list.append([language_code,confidence])
    for langauge in lingua[0]: #convert to list
        language_code,confidence = str(langauge).split(":")
        combined_list.append([language_code,confidence])

    merged_list = datamanager.mergeResultList(combined_list,mode) #merge lists

    langauges = []

    for language in merged_list: #convert back to list of strings
        langauges.append(str(language[0] + ":" + str(language[1])))

    return langauges


#get the most likely languages
def getLanguage(text,mode=2):
    results = []
    for alphabet in text:
        try: #list with languages and number of word
            if mode == 0: #langdetect
                print ("detecting with langdetect")
                languages = [detect_langs(alphabet[0]),alphabet[1]]
            elif mode == 1: #lingua
                print("detecting with lingua")
                languages = [detectLanguageLingua(alphabet[0]),alphabet[1]]
            elif mode == 2: #combination
                print("detecting combined")
                languages_lang = [detect_langs(alphabet[0]),alphabet[1]]
                languages_lingua = [detectLanguageLingua(alphabet[0]), alphabet[1]]
                languages = [combineLanguageResults(languages_lang,languages_lingua),alphabet[1]]

            results.append(languages)
        except:
            results.append(None)

    return results


#get the positions of text in the image
def locateText(bounding_boxes,x=640,y=640):

    positions=[False,False,False,False,False,False,False,False,False]#upleft.midleft,downleft,upmid and so on
    rectangles = []
    width_step = x / 3
    height_step = y / 3

    for w in range(3): #iterate over all position squares
        current_width = w * width_step
        for h in range(3):
            current_height = h * height_step
            rectangles.append(box(current_width, current_height,current_width + width_step, current_height + height_step)) #create a box for each square
    for bounding in bounding_boxes:
        middle_point_x = (bounding[0][0] + bounding[1][0] + bounding[2][0] + bounding[3][0]) / 4
        middle_point_y = (bounding[0][1] + bounding[1][1] + bounding[2][1] + bounding[3][1]) / 4
        middle_point = Point(middle_point_x,middle_point_y) #create a point for each bounding box middle point
        for index,rectangle in enumerate (rectangles):
            if (rectangle.contains(middle_point)): #if the middle point is inside a square
                positions[index] = True #set the respective position in the array to true

    return positions

#zoom in on the image parts where text was located by requesting a new image via the API
def zoomText(image,positions,new_fov,heading_scale=3,pitch_scale=3,debug=False):

    images = []

    for index,position in enumerate(positions): #for all squares in position

        if position == True: #request new image with fitting parameters when value true

            print("zooming in")

            if index in [0,1,2]: #left squares
                new_heading = float(image["heading"]) - float(image["fov"]) / heading_scale
            elif index in [3,4,5]: #middle squares
                new_heading = float(image["heading"])
            elif index in [6,7,8]: #right squares
                new_heading = float(image["heading"]) + float(image["fov"]) / heading_scale

            if index in [0,3,6]: #upper squares
                new_pitch = float(image["pitch"]) + float(image["fov"]) / pitch_scale

            elif index in  [1,4,7]:#middle squares
                new_pitch = float(image["pitch"])

            elif index in  [2,5,8]: #bottom squares
                new_pitch=float (image["pitch"]) - float (image["fov"]) / pitch_scale

            zoomed_image = datamanager.getImageWithDict(coordinates=image["coordinates"],heading=new_heading,fov=new_fov,pitch=new_pitch,debug=debug,source="default")
            images.append(zoomed_image)

    return images

#get the bounding boxes of detected text
def getBoundingBoxes(results):
    boxes = []
    for result in results:
        for text in result[2]:
            boxes.append(text[0]) #just take the bounding boxes
    return boxes

#get the text without confidence values
def getText(results):
    words = []
    #print (results)
    for text in results[2]:
            words.append(text[1]) #just take the text

    return words

#check if any popular place names are mentioned
def checkForPlaceNames(text):

    results = []
    for alphabet in text:
        places = GeoText(alphabet[0]) #check for places
        countries = places.country_mentions #get countries of those places
        for country in countries:
            country_code = pycountry.countries.get(alpha_2=country).alpha_3
            results.append([country_code,min(1,(countries.get(country)/2))]) #increased likelihood when numerous mentions

    return results

#check if any country domains are mentioned
def checkForDomainNames(text):

    results = []
    countries = datamanager.getAllCountriesWithExternalData()

    for country in countries:
        facts = datamanager.loadCountryFacts(country)
        for alphabet in text:
            words = alphabet[0].split(" ") #split text into words
            for word in words:
                if word == facts["domain"]: #check if word equals one of the specified domains
                    results.append([country,0.5]) #increase likelihood for the respective country when true

    return datamanager.mergeResultList(results) #merge the list so that no country is more often included than once

#get the countries for the detected languages
def languageToCountry(languages):

    results = []
    countries = datamanager.getAllCountriesWithExternalData()

    for country in countries: #for each country
        facts = datamanager.loadCountryFacts(country)
        for alphabet in languages:
            if alphabet:
                number = alphabet[1]
                for language in alphabet[0]: #for each detected language
                    name,confidence = str(language).split(":")
                    if name in facts["languages"]: #check if it is mentioned in the fact sheet
                        score = min (1,float(confidence) * (float(number) / 10)) #score based on language confidence and number of words
                        results.append([country,score])

    return datamanager.mergeResultList(results) #merge the list so that no country is more often included than once

#detect text, get its most likely languages, check for place names and domains and increase likelihood for countries with the detected languages
def analyseLanguage(images,zoom_min = 0, zoom_max=0.4,alphabets=[0,1,2,3,4,5,6,7,8,9],lang_mode=2,image_filter=True):

    used_alphabets = {}
    print ("searching for text")
    for image in images:
        detected_alphabets = getMostConfident(image,alphabets,image_filter) #get the two alphabets with the highest confidence value
        #print (detected_alphabets)
        if (detected_alphabets[0][1] > zoom_min and detected_alphabets[0][1] < zoom_max): #if highest confidence is within a certain range
            print ("detected text, zooming for better readability")
            boxes = getBoundingBoxes(detected_alphabets) #get bounding boxes of detected text
            positions = locateText(boxes) #get the position of the bounding box inside the image
            zoomed_images=zoomText(image,positions,25) #zoom on those positions
            for zoomed_image in zoomed_images:
                detected_alphabets = getMostConfident(zoomed_image,alphabets)
            for writing in detected_alphabets: #for each alphabet
                if (writing[1] > zoom_max): #when confidence is high enough
                    try: #form new dictionary of used alphabets with alphabet number as keys and detected text as words
                        used_alphabets[str(writing[0])] = used_alphabets[str(writing[0])] + getText(writing)
                    except:
                        used_alphabets[str(writing[0])] = getText(writing)

        if detected_alphabets[0][2] != []: #if there actually is text detected
            for writing in detected_alphabets:
                if (writing[1] > zoom_max): #when confidence is high enough
                    try: #form new dictionary of used alphabets with alphabet number as keys and detected text as words
                        used_alphabets[str(writing[0])] =  used_alphabets[str(writing[0])] + getText(writing)
                    except:
                        used_alphabets[str(writing[0])] = getText(writing)

    print ("found alphabets:",list(used_alphabets.keys()))

    text = getCompleteText(used_alphabets) #form a single string out of all words
    print ("found text: ", text)

    languages = getLanguage(text,lang_mode) #get the most likely languages
    print ("found languages:", languages)
    countries_lang = languageToCountry(languages) #get the countries using these languages

    countries_places = checkForPlaceNames(text) #check if place names are mentioned and get the respective countries
    print ("found place names: ",countries_places)

    countries_domains = checkForDomainNames(text) #check if the text contains domain names and get the respective countries
    print("found domain names: ", countries_domains)

    countries_lang.extend(countries_places) #add the countries with mentioned place names to the countries by language
    countries_lang.extend(countries_domains) #add the countries with mentioned domains to the countries by language

    countries_lang = datamanager.mergeResultList(countries_lang) #merge the list so that no country is more often included than once

    print ("possible countries based on language and text: ", countries_lang)

    return countries_lang


#initialise text readers when imported by other module
initTextReaders()

