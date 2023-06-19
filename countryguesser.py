import random
import datamanager
import imageeditor
import sundetector
import textandlanguagedetector
import colordetector
import contextdetector
import objectdetector

#calculate the propabilty scores for all countries
def calculateTotalCountryScore(sun=[],language=[],color=[],context=[],license=[],density=[],mode=[0,6],random_weightings=None,standard_weightings=None):

    countries = datamanager.getAllCountriesWithExternalData()

    results = []

    for country in countries:

        sun_score = 0
        language_score = 0
        color_score = 0
        context_score = 0
        license_score = 0
        density_score = 0


        for s in sun:
            if country in s:
                sun_score = s[1]

        for l in language:
            if country in l:
                language_score = l[1]

        for c in color:
            if country in c:
                color_score = c[1]

        for c in context:
            if country in c:
                context_score = c[1]

        for l in license:
            if country in l:
                license_score = l[1]

        for d in density:
            if country in d:
                density_score = d[1]

        if mode[0] == 0:
            score = weightScores(mode[1],sun_score,language_score,color_score,context_score,license_score,density_score,countries,random_weightings,standard_weightings)
        if mode[0] == 1:
            score = weightPlacings(mode[1],sun,language,color,context,license,density,country,countries,random_weightings,standard_weightings)
        if mode[0] == 2:
            score = weightMergedPlacings(mode[1],sun,language,color,context,license,density,language_score,color_score,context_score,license_score,density_score,country,countries,random_weightings,standard_weightings)
        if mode[0] == 3:
            score = weightScoresNormalized(mode[1],sun,language,color,context,license,density,sun_score,language_score,color_score,context_score,license_score,density_score,random_weightings,standard_weightings)
        if mode[0] == 4:
            score = weightScoresNormalizedDataset(mode[1],sun,language,license,density,sun_score,language_score,color_score,context_score,license_score,density_score,random_weightings,standard_weightings)

        results.append([country, score, sun_score, language_score, color_score, context_score, license_score, density_score])


    results.sort(key=lambda x: x[1], reverse=True)

    return results


#weightings based on scores
def weightScores(mode,sun_score,language_score,color_score,context_score,license_score,density_score,countries,random_weightings,standard_weightings):

    score = 0

    if mode == 0: #best practice
        score = sun_score + language_score + color_score * 2 + context_score * len(countries) * 2 + license_score * 0.2 + density_score * len(countries)  # /2 if not object list
    elif mode == 1: #unweighted
        score = sun_score + language_score + color_score + context_score + license_score + density_score
    elif mode == 2: #average placing in test dataset
        score = sun_score + language_score + color_score / 12.5 + context_score / (len(countries) * 2 / 18) + license_score / 15 + density_score / (len(countries) / 31.5)
    elif mode == 3: #highest average score in test dataset
        score = sun_score + language_score + color_score / 0.87 + context_score / 0.034 + license_score / 0.51 + density_score / 0.011
    elif mode == 4: #highest average score in test dataset and best practice
        score = sun_score + language_score + 2 * (color_score / 0.87) + 2 * (context_score / 0.034) + 0.2 * (license_score / 0.51) + density_score / 0.011
    elif mode == 5: #average scores of searched countries and their ratios to the average highest scores in the test dataset
        score = sun_score + language_score * 0.77 + (color_score / 0.87) * 0.95 + (context_score / 0.034) * 0.62 + (license_score / 0.51) * 0.94 + (density_score / 0.011) * 0.3
    elif mode == 6:  #automated testing (best practice, 0.9 - 1.1)
        score = 0.97 * sun_score + 0.49 * language_score + 17.8 * color_score + context_score * len(countries) * 1.58 + license_score * 0.3 + density_score * len(countries) * 0.9
    elif mode == 7: #automated testing (no weightings, 0.9 - 1.1)
        score = 0.39 * sun_score + 0.36 * language_score + 11 * color_score + context_score * len(countries) * 0.97 + license_score * 0.54 + density_score * len(countries) * 0.56
    elif mode == 8: #automated testing (no weightings, 0 - 2)
        score = 0.002 * sun_score + 0.07 * language_score + 3.8 * color_score + context_score * len(countries) * 0.47 + license_score * 0.26 + density_score * len(countries) * 0.14
    elif mode == -1: #for automated setting
        score = sun_score * standard_weightings[0] * random_weightings[0] + language_score * standard_weightings[1] * random_weightings[1] + color_score * standard_weightings[2] * random_weightings[2] + context_score * standard_weightings[3] * random_weightings[3] + license_score * standard_weightings[4] * random_weightings[4] + density_score * standard_weightings[5] * random_weightings[5]

    return score

#weightings based on placings
def weightPlacings(mode,sun,language,color,context,license,density,country,countries,random_weightings,standard_weightings):

    score = 0

    if mode == 0: #unweighted
        score = 6 * len(countries) - ((getCountryPlacing(country, sun) or 0) + (getCountryPlacing(country, language) or 0) + getCountryPlacing(country, color) + getCountryPlacing(country, context) + (getCountryPlacing(country, license) or 0) + (getCountryPlacing(country, density) or 0))
    elif mode == 1: #best practice
        score = 6 * len(countries) - ((getCountryPlacing(country, sun) or 0) + (getCountryPlacing(country, language) or 0) + 2 * getCountryPlacing(country, color) + 2 * getCountryPlacing(country, context) + 0.25 * (getCountryPlacing(country, license) or 0) + 0.5 * (getCountryPlacing(country, density) or 0))
    elif mode == 2: #average placings in test dataset
        score = 6 * len(countries) - ((getCountryPlacing(country, sun) or 0) + (getCountryPlacing(country, language) or 0) / 2 + getCountryPlacing(country, color) / 25 + getCountryPlacing(country, context) / 18 + (getCountryPlacing(country, license) or 0) / 3 + (getCountryPlacing(country, density) or 0) / 31.5)
    elif mode == 3: #average score of searched countries and their ratio to the highest avergae score in test dataset
        score = 6 * len(countries) - ((getCountryPlacing(country, sun) or 0) + (getCountryPlacing(country, language) or 0) / 0.77 + getCountryPlacing(country,color) / 0.95 + getCountryPlacing(country, context) / 0.62 + (getCountryPlacing(country, license) or 0) / 0.94 + (getCountryPlacing(country, density) or 0) / 0.34)
    elif mode == -1: #for automated setting
        score = 6 * len(countries) - ((getCountryPlacing(country,sun) or 0) * standard_weightings[0] * random_weightings[0] + (getCountryPlacing(country,language) or 0) * standard_weightings[1] * random_weightings[1] + getCountryPlacing(country,color) * standard_weightings[2] * random_weightings[2] + getCountryPlacing(country,context) * standard_weightings[3] * random_weightings[3] + (getCountryPlacing(country,license) or 0) *  standard_weightings[4] * random_weightings[4] + (getCountryPlacing(country,density) or 0) * standard_weightings[5] * random_weightings[5])

    return score

#weightings based on merged placings
def weightMergedPlacings(mode,sun,language,color,context,license,density,language_score,color_score,context_score,license_score,density_score,country,countries,random_weightings,standard_weightings):

    score = 0

    if mode == 0: #unweighted
        score = 6 * len(countries) - ((getMergedPlacing(country, sun) or 0) + (getMergedPlacing(country, language) or 0) + getMergedPlacing(country, color) + getCountryPlacing(country, context) + (getMergedPlacing(country, license) or 0) + (getMergedPlacing(country, density) or 0))
    elif mode == 1: #best practice
        score = 6 * len(countries) - ((getMergedPlacing(country, sun) or 0) + (getMergedPlacing(country, language) or 0) + 2 * getMergedPlacing(country, color) + 2 * getMergedPlacing(country, context) + 0.25 * (getMergedPlacing(country, license) or 0) + 0.5 * (getMergedPlacing(country, density) or 0))
    elif mode == 2: #average placings in test dataset
        score = 6 * len(countries) - ((getMergedPlacing(country, sun) or 0) + (getMergedPlacing(country, language) or 0) / 2 + getMergedPlacing(country, color) / 25 + getMergedPlacing(country, context) / 18 + (getMergedPlacing(country, license) or 0) / 3 + (getMergedPlacing(country, density) or 0) / 31.5)
    elif mode == 3: #average scores of searched countries and their ratios to the average highest score in test dataset
        score = 6 * len(countries) - ((getMergedPlacing(country, sun) or 0) + (getMergedPlacing(country, language) or 0) / 0.77 + getMergedPlacing(country,color) / 0.95 + getMergedPlacing(country, context) / 0.62 + (getMergedPlacing(country, license) or 0) / 0.94 + (getMergedPlacing(country, density) or 0) / 0.34)
    elif mode == 4: #highest average scores in test dataset
        score = 6 * len(countries) - ((getMergedPlacing(country, sun) or 0) + (getMergedPlacing(country, language) or 0) / max(language_score,0.01) + 2 * (getMergedPlacing(country, color) / (color_score / 0.87)) + 2 * (getMergedPlacing(country, context) / (context_score / 0.034)) + 0.25 * ((getMergedPlacing(country, license) or 0) / (max(license_score, 0.01) / 0.51)) + 0.5 * ((getMergedPlacing(country, density) or 0) / (max(density_score, 0.01) / 0.012)))
    elif mode == 5: #second best result of automated testing (best practice,0-2,no checkpoints)
        score = 6 * len(countries) - ((getMergedPlacing(country, sun) or 0) * 1.58 + (getMergedPlacing(country, language) or 0) * 0.8 + getMergedPlacing(country,color) * 2.82 + getCountryPlacing(country, context) * 3.94 + (getMergedPlacing(country, license) or 0) * 0.1725 + (getMergedPlacing(country, density) or 0) * 0.775)
    elif mode == 6: #best result of automated testing (best practice,0-2,no checkpoints)
        score = 6 * len(countries) - ((getMergedPlacing(country, sun) or 0) * 1.45 + (getMergedPlacing(country, language) or 0) * 0.08 + getMergedPlacing(country,color) * 2.16 + getCountryPlacing(country, context) * 3.04 + (getMergedPlacing(country, license) or 0) * 0.285 + (getMergedPlacing(country, density) or 0) * 0.58)
    elif mode == 7: #automated testing (no weightings, 0-2)
        score = 6 * len(countries) - ((getMergedPlacing(country, sun) or 0) * 0.36 + (getMergedPlacing(country, language) or 0) * 0.038 + getMergedPlacing(country,color) * 9.99 + getCountryPlacing(country, context) * 13.96 + (getMergedPlacing(country, license) or 0) * 0.0006 + (getMergedPlacing(country, density) or 0) * 2.73)
    elif mode == 8: #automated testing (no weightings, 0.9-1.1)
        score = 6 * len(countries) - ((getMergedPlacing(country, sun) or 0) * 0.76 + (getMergedPlacing(country, language) or 0) * 0.88 + getMergedPlacing(country,color) * 1.61 + getMergedPlacing(country, context) * 2.26 + (getMergedPlacing(country, license) or 0) + 1.36 + (getMergedPlacing(country, density) or 0) * 0.52)
    elif mode == 9: #best practice and total number of countries
        score = 6 * len(countries) - ((getMergedPlacing(country, sun) or 0) / len(countries) + (getMergedPlacing(country, language) or 0) / len(countries) + 2 * getMergedPlacing(country,color) / len(countries) + 2 * getMergedPlacing(country, context) / len(countries) + 0.25 * (getMergedPlacing(country, license) or 0) / len(countries) + 0.5 * (getMergedPlacing(country, density) or 0) / len(countries))
    elif mode == 10: #total numbers of countries and automated testing (no weightings, 0.9 - 1.1)
        score = 6 * len(countries) - ((getMergedPlacing(country, sun) or 0) / len(countries) + ((getMergedPlacing(country, language) or 0) / len(countries)) * 1.19 + (getMergedPlacing(country, color) / len(countries)) * 3.28 + (getMergedPlacing(country, context) / len(countries)) * 4.74 + ((getMergedPlacing(country, license) or 0) / len(countries)) + 0.76 + ((getMergedPlacing(country, density) or 0) / len(countries)) * 0.71)
    elif mode == -1: #for automated setting
        score = 6 * len(countries) - ((getMergedPlacing(country,sun) or 0) * standard_weightings[0] * random_weightings[0] + (getMergedPlacing(country,language) or 0) * standard_weightings[1] * random_weightings[1] + getMergedPlacing(country,color) * standard_weightings[2] * random_weightings[2] + getCountryPlacing(country,context) * standard_weightings[3] * random_weightings[3] + (getMergedPlacing(country,license) or 0) *  standard_weightings[4] * random_weightings[4] + (getMergedPlacing(country,density) or 0) * standard_weightings[5] * random_weightings[5])

    return score

#weightings based on scores normalised with maximum value as one
def weightScoresNormalized(mode,sun,language,color,context,license,density,sun_score,language_score,color_score,context_score,license_score,density_score,random_weightings,standard_weightings):

    score = 0

    if len(sun) > 0:
        sun_score = sun_score / max(sun, key=lambda x: x[1])[1]
    if len(language) > 0:
        language_score = language_score / max(language, key=lambda x: x[1])[1]
    if len(license) > 0:
        license_score = license_score / max(license, key=lambda x: x[1])[1]
    if len(density) > 0:
        density_score = density_score / max(density, key=lambda x: x[1])[1]

    color_score = color_score / max(color, key=lambda x: x[1])[1]
    context_score = context_score / max(context, key=lambda x: x[1])[1]

    if mode == 0: #no weightings
        score = sun_score + language_score + color_score + context_score + license_score + density_score
    elif mode == 1: #best practice
        score = sun_score + language_score + 2 * color_score + 2 * context_score + 0.25 * license_score + 0.5 * density_score
    elif mode == 2: #average placing in test dataset
        score = sun_score + language_score / 2 + color_score  / 25 + context_score  / 18 + license_score / 3 + density_score / 31.5
    elif mode == 3: #automated testing (no weightings, 0.9 - 1.1)
        score = 0.54 * sun_score + 0.47 * language_score + color_score  * 8.25 + context_score * 2.71 + license_score * 0.3 + density_score * 0.4
    elif mode == -1: #for automated setting
        score = sun_score * standard_weightings[0] * random_weightings[0] + language_score * standard_weightings[1] * random_weightings[1] + color_score * standard_weightings[2] * random_weightings[2] + context_score * standard_weightings[3] * random_weightings[3] + license_score * standard_weightings[4] * random_weightings[4] + density_score * standard_weightings[5] * random_weightings[5]

    return score

#weightings based on scores normalised with maximum value from test dataset as one
def weightScoresNormalizedDataset(mode,sun,language,license,density,sun_score,language_score,color_score,context_score,license_score,density_score,random_weightings,standard_weightings):

    score = 0

    if len(sun) > 0:
        sun_score = sun_score / 1
    if len(language) > 0:
        language_score = language_score / 1
    if len(license) > 0:
        license_score = license_score / 0.51
    if len(density) > 0:
        density_score = density_score / 0.012

    color_score = color_score / 0.87
    context_score = context_score / 0.034

    if mode == 0: #no weightings
        score = sun_score + language_score + color_score + context_score + license_score + density_score
    elif mode == 1: #best practice
        score = sun_score + language_score + 2 * color_score + 2 * context_score + 0.25 * license_score + 0.5 * density_score
    elif mode == 2: #automated testing (no weightings, 0.9 - 1.1)
        score = 0.41 * sun_score + 0.46 * language_score + 7.16 * (color_score) + 2.88 * (context_score) + 0.17 * license_score + 0.29 * density_score
    elif mode == -1: #for automated setting
        score = sun_score * standard_weightings[0] * random_weightings[0] + language_score * standard_weightings[1] * random_weightings[1] + color_score * standard_weightings[2] * random_weightings[2] + context_score * standard_weightings[3] * random_weightings[3] + license_score * standard_weightings[4] * random_weightings[4] + density_score * standard_weightings[5] * random_weightings[5]

    return score


#make a guess for a country
def makeGuess(sky,horizon_one,horizon_two,mask=[1,1,1,1,1,1],alphabets=[0,1,2,3,4,5,6,7,8,9],country=None,zoom_allowed=True,mode=[0,6],file=None):

    sun = []
    language = []
    color = []
    context = []
    license = []
    density = []

    if horizon_two == None:
        horizon_two = horizon_one

    if mask[0]: #sun
        print("guessing sun with images with fovs: ",sky[0]["fov"])
        sun = sundetector.analyseSun(sky)
    if mask[1]: #language and text
        print ("guessing language with images with fov ",horizon_one[0]["fov"])
        if zoom_allowed:
            language = textandlanguagedetector.analyseLanguage(horizon_one,alphabets=alphabets)
        else:
            language = textandlanguagedetector.analyseLanguage(horizon_one, alphabets=alphabets,zoom_min=1,zoom_max=1)

    print ("guessing remaining elements with images with fov ", horizon_two[0]["fov"])
    if mask[2]: #color
        color = colordetector.analyseHistograms(horizon_two,mode=0,number=300)
        if country:
            print ("color rank: ", getCountryPlacing(country,color))

    if mask[3]: #context
        context = contextdetector.analyseWordLists(horizon_two,model="conceptual",number=300)
        if country:
            print ("context rank: ", getCountryPlacing(country,context))

    if mask[4]: #license
        license = objectdetector.analyseObjects(horizon_two,zoom_allowed=zoom_allowed)

    if mask[5]: #general object detection
        density = objectdetector.analyseGeneralObjectDetection(horizon_two,mode=0,number=300)
        if country:
            print ("density rank: ", getCountryPlacing(country,density))

    if file != None:
        datamanager.saveResults(sun,language,color,context,license,density,country,file)

    countries = calculateTotalCountryScore(sun,language,color,context,license,density,mode)
    #countries.sort(key=lambda x: x[1],reverse=True)

    print("best results:", countries[0:2])
    #print("complete results:", countries)

    return countries

#collect images for sky and horizon
def collectImages(coordinates,fovs=[45,90],debug=True):

    print ("collecting images")

    sky = datamanager.scanSky(coordinates,debug=debug)

    horizon_two=None
    if len(fovs) == 2:
        horizon_one = datamanager.getPanoramaAtPitch(coordinates,0,fov=fovs[0],debug=debug)
        horizon_two = datamanager.getPanoramaAtPitch(coordinates,0,fov=fovs[1],debug=debug)
    else:
        horizon_one = datamanager.getPanoramaAtPitch(coordinates,0,fov=fovs[0],debug=debug)

    return sky,horizon_one,horizon_two

#collect images and make a guess
def collectAndGuess(coordinates,mask=[1,1,1,1,1,1],alphabets=[0,1,2,3,4,5,6,7,8,9],country=None):

    if mask[1]:
        sky, horizon_one, horizon_two = collectImages(coordinates,fovs=[45,90])
    else:
        sky, horizon_one, horizon_two = collectImages(coordinates,fovs=[90])

    return makeGuess(sky,horizon_one,horizon_two,mask,alphabets,country)

#get the placing of a country in a result list
def getCountryPlacing(country, result):
    place = 0
    for res in result:
        place = place + 1
        if res[0] == country:
            return place

#get the placing of a country with same confidence scores on same placing
def getMergedPlacing(country,result):

    if len(result) > 0:

        current_rank = 1
        current_score = result[0][1]

        for res in result:
            if res[1] != current_score:
                current_rank = current_rank + 1
                current_score = res[1]
            if res[0] == country:
                return current_rank

    return None

#get a sub set of recored results
def getResultsSubSet(results,start=0,ending=None):

    if not ending:
        ending = len(results)

    return results[start:ending]

#get statistics about recorded results
def getAverageModulePlacings(file="last_results.json",start=0,ending=None):

    results = datamanager.loadJSON(file)

    placings={}
    highscores={}
    searched={}

    results = getResultsSubSet(results,start,ending)

    for index, res in enumerate(results):

        country = res["country"]
        res["complete"]=calculateTotalCountryScore(res["sun"],res["language"],res["color"],res["context"],res["license"],res["density"],mode=[0,6])
        modules = ["complete","sun","language","color","context","license","density"]

        for module in modules:
            placing = getMergedPlacing(country,res[module])
            if module not in placings:
                placings[module] = []
                highscores[module]=[]
                searched[module]=[]

            placings[module].append(placing)

            if len(res[module]) > 0:
                highscores[module].append(res[module][0][1])
                placing = getCountryPlacing(country,res[module])
                if placing:
                    searched[module].append(res[module][placing-1][1])

    for module in modules:
        print ("placings ",module)
        print (placings[module])
        print ("highest scores ",module)
        print (highscores[module])
        print ("scores of searched answers", module)
        print (searched[module])

#get scores for several weightings and recorded results
def testWeightings(file="last_results.json",random_weightings=None,standard_weightings=None,start=0,ending=None,mode=0):

    results = datamanager.loadJSON(file)

    placings={}

    results = getResultsSubSet(results,start,ending)

    guesses = len(results)

    print (guesses)

    for index,res in enumerate (results):

        sun = res["sun"]
        language = res["language"]
        color = res["color"]
        context=res["context"]
        license=res["license"]
        density=res["density"]

        country=res["country"]

        if not random_weightings: #iterate over all weightings

            for method in range(5):
                index = 0
                check_score = 1
                while check_score != 0:
                    score = calculateTotalCountryScore(sun, language, color, context, license, density,mode=[method, index])
                    check_score = score[0][1]
                    placing = getCountryPlacing(country,score)

                    if check_score != 0:
                        if (str(method) + ":" + str(index)) not in placings:
                            placings[str(method) + ":" + str(index)] = placing
                        else:
                            placings[str(method) + ":" + str(index)] = placings[str(method) + ":" + str(index)] + placing

                    index = index + 1

        else: #get placing for the automated setting of weightings
            score = calculateTotalCountryScore(sun, language, color, context, license, density, mode=[mode,-1],random_weightings=random_weightings,standard_weightings=standard_weightings)
            placing = getCountryPlacing(country, score)
            if "automating" in placings:
                placings["automating"] = placings["automating"] + placing
            else:
                placings["automating"] = placing

    print ("placings by mode:")
    for mode in placings:
        print (mode,placings[mode] / guesses)

    if "automating" in placings: #only return a value if automated setting
        return placings["automating"] / guesses


#set weightings automaticaly
def adaptWeightingsRandom(attempts=1000,best_ranking=110,best_weightings=[1,1,1,1,1,1],standard_weightings=[1,1,1,1,1,1],random_range=[0.9,1.1],mode=0):

    for index in range (attempts):

        random_weightings = [round(random.uniform(random_range[0], random_range[1]), 2) for index in range(6)]

        ranking = testWeightings(random_weightings=random_weightings,standard_weightings=standard_weightings,mode=mode)
        print (ranking)

        if ranking < best_ranking: #if ranking with new weightings is better
            print ("new best ranking")
            print ("weightings: ", random_weightings)
            best_ranking = ranking
            best_weightings = random_weightings
            for index, weighting in enumerate(standard_weightings): #new weightings based on so far best
                standard_weightings[index] = best_weightings[index] * weighting
            print ("standard weightings", standard_weightings)

    return best_ranking,best_weightings,standard_weightings

#test the whole system
def testRun(number=1,mask=[1,1,1,1,1,1],alphabets=[0,1,2,3,4,5,6,7,8,9],dataset=None,full_dataset=True,file="last_results"):

    countries = datamanager.getAllCountriesWithExternalData()
    average_place = 0

    used_coordinates = []

    if dataset: #test with local dataset
        local_coordinates = datamanager.getDataSetCoordinates(dataset)
        if full_dataset:
            number = len(local_coordinates)

    for i in range(number):
        if not dataset:
            country = random.choice(countries)
            point = datamanager.getPointInCountry(datamanager.getCountryNumber(country))
            print("Guessing for", country,point)
            result = collectAndGuess(point, mask, alphabets, country)
        else:
            point = random.choice(local_coordinates)
            local_coordinates.remove(point)
            point = datamanager.unpackCoordinates(point)
            point = tuple([float(coordinate) for coordinate in point])
            country = datamanager.getCountryofPoint(point)[1]
            print("Guessing for", country,point)
            sky, horizon_one,horizon_two = datamanager.loadFromTestDataset(dataset,point)
            result = makeGuess(sky, horizon_one, horizon_two, mask, alphabets, country,file=file)

        used_coordinates.append(point)

        place = getCountryPlacing(country,result)
        print("country placing: ", place)

        average_place = average_place + place


    print ("coordinates",used_coordinates)

    print ("average placing", average_place / number)

    return average_place / number

#test the sun detector module
def testSunDetector(hemi,number=50):
    countries = datamanager.getCountriesByHemishphere(hemi)
    if hemi == "north":
        other_countries = datamanager.getCountriesByHemishphere("south")
    else:
        other_countries = datamanager.getCountriesByHemishphere("north")
    correct_guesses = 0
    wrong_guesses = 0
    for i in range(number):
        country = random.choice(countries)
        print("Guessing for", country)
        coordinates = datamanager.getPointInCountry(datamanager.getCountryNumber(country))
        images = datamanager.scanSky(coordinates,debug=False)
        results = sundetector.analyseSun(images)
        results = [res[0] for res in results]
        if country in results:
            correct_guesses = correct_guesses + 1
        elif other_countries[0] in results:
            wrong_guesses = wrong_guesses + 1

    unclear = number - (correct_guesses + wrong_guesses)

    print (hemi,correct_guesses,wrong_guesses,unclear)

    return correct_guesses,wrong_guesses,unclear


#test the text and language detector module
def testTextandLanguageGuesser(countries=None,number = 100,modes = [False,False,True]):

    right_guesses = [ [],[],[] ]
    wrong_guesses = [ [],[],[] ]

    not_guessed = [0,0,0]

    if countries == None:
        countries = datamanager.getAllCountriesWithExternalData()

    for index in range(number):
        country = random.choice(countries)

        coordinates = False

        while not coordinates:
            coordinates = datamanager.getPointInCountry(datamanager.getCountryNumber(country))

        images = datamanager.getPanoramaAtPitch(coordinates,fov=45,pitch=0)

        guessed_countries = []

        if modes[0]:
            guessed_countries.append( textandlanguagedetector.analyseLanguage(images,lang_mode=0))
        if modes[1]:
            guessed_countries.append(textandlanguagedetector.analyseLanguage(images,lang_mode=1))
        if modes[2]:
            guessed_countries.append(textandlanguagedetector.analyseLanguage(images,lang_mode=2))

        for index,guess in enumerate (guessed_countries):
            if guess and len(guess) > 0:
                for guessed_country in guess:
                    print(country, guessed_country[0])
                    if country == guessed_country[0]:
                        right_guesses[index].append(guessed_country)
                    else:
                        wrong_guesses[index].append(guessed_country)
            else:
                not_guessed[index] = not_guessed[index] + 1

    average_confidences = [0,0,0]
    other_confidences = [0,0,0]

    for index,mode in enumerate (right_guesses):
        for country in  mode:
            average_confidences[index] = average_confidences[index] + country[1]

    for index,mode in enumerate (wrong_guesses):
        for country in  mode:
            other_confidences[index] = other_confidences[index] + country[1]

    for index,mode in enumerate (right_guesses):
        if len (mode) > 0:
            average_confidences[index] = average_confidences[index] / len (mode)
        else:
            average_confidences[index] = 0

    for index,mode in enumerate (wrong_guesses):
        if len (mode) > 0:
            other_confidences[index] = other_confidences[index] / len (mode)
        else:
            other_confidences[index] = 0

    for index in range (len (right_guesses)):

        print ("mode:", index)

        print("right guesses :", right_guesses[index])
        print("wrong guesses :", wrong_guesses[index])

        print("right: ", len(right_guesses[index]))
        print("percentage: ", len(right_guesses[index]) / number)

        print("no guess possible: ", not_guessed[index])
        print("percentage of actual guesses", len(right_guesses[index]) / (number - not_guessed[index]))

        print("average confidence: ", average_confidences[index])

        print("wrong", len(wrong_guesses[index]))

        print("other confidence: ", other_confidences[index])


#test the different distance metrics for images from same and different countries
def testErrorMetrics(number=100):
    errors = [[-1.0,-1.0,-1.0],[-1.0,-1.0,-1.0]]
    countries = datamanager.getAllCountriesWithExternalData()
    for index in range(number):
        country_one = random.choice(countries)
        country_two = country_one
        coordinates = datamanager.getPointInCountry(datamanager.getCountryNumber(country_one))
        image_one = datamanager.getImageWithDict(coordinates,heading=random.randrange(0,360))
        #image_one = datamanager.getImageWithDict(coordinates)
        if (index%2 != 0):
            while country_one == country_two:
                country_two = random.choice(countries)

        coordinates = datamanager.getPointInCountry(datamanager.getCountryNumber(country_two))
        image_two = datamanager.getImageWithDict(coordinates,heading=random.randrange(0,360))
        #image_two = datamanager.getImageWithDict(coordinates)

        errors[index % 2][0] = errors[index % 2][0] +  (1 - imageeditor.compareImagesAvergeDifferenceRGB(image_one,image_two))

        errors[index % 2][1] = errors[index % 2][1] + (1 - imageeditor.compareImagesNormalizedRootMeanSquareError(image_one,image_two))

        errors[index % 2][2] = errors[index % 2][2] +  imageeditor.compareImagesStructuralSimilarity(image_one,image_two)

    for differences in errors:
        for index,method in enumerate (differences):
            differences[index] = method / (number / 2)

    print ("images from same country:",errors[0],"images from different countries:",errors[1])

    return errors

#test different methods of the color detection module
def testColorDetectionMethods(country_number=10,image_number=100,panoramas=False):

    countries = datamanager.getAllCountriesWithExternalData()

    if country_number == 0:
        country_number = len(countries)
    else:
        countries = random.sample(countries,country_number)

    print (country_number)

    country_data = [{},{},{},{},{},{},{},{},{}]

    for country in countries:
        print ("generating for ", country)
        images = datamanager.loadAllCountryImages(country=country,max=100,fovs=[90])

        print ("average images")
        country_data[0][country] = datamanager.packImageDict(imageeditor.mergeImagesRGBNumpy(images,sharpen=True),(0,0),0,0,0)
        country_data[1][country] = datamanager.packImageDict(imageeditor.mergeImagesRGBNumpy(images),(0,0),0,0,0)

        if not panoramas:
            print("cluster")
            country_data[2][country] = imageeditor.getImageCluster(images,sharpen=True)
            country_data[3][country] = imageeditor.getImageCluster(images)

            print("alternative cluster")
            country_data[4][country] = imageeditor.clusterImageCluster(images,weighted=False)
            country_data[5][country] = imageeditor.clusterImageCluster(images,weighted=True)

            print("representative image")
            country_data[6][country] = imageeditor.mostSimilarImage(images)

        print("histograms")
        country_data[7][country] = imageeditor.getAverageHistogram(images)
        country_data[8][country] = country_data[7][country]

    placings = [0] * len(country_data)

    for image in range(image_number):

        image_data = [0] * len(country_data)

        image_country = random.choice(countries)

        print ("image country ",image_country)

        coordinates = False

        while not coordinates:
            coordinates = datamanager.getPointInCountry(datamanager.getCountryNumber(image_country))

        if not panoramas:
            images = [datamanager.getImageWithDict(coordinates)]
        else:
            images = datamanager.getPanoramaAtPitch(coordinates,pitch=0)


        image_data[0]=datamanager.packImageDict(imageeditor.mergeImagesRGBNumpy(images),(0,0),0,0,0)
        image_data[1]=datamanager.packImageDict(imageeditor.mergeImagesRGBNumpy(images,sharpen=False),(0,0),0,0,0)

        if not panoramas:

            image_data[2]=imageeditor.getImageCluster(images)
            image_data[3]=imageeditor.getImageCluster(images,sharpen=False)

            image_data[4]=imageeditor.clusterImageCluster(images, weighted=False)
            image_data[5]=imageeditor.clusterImageCluster(images, weighted=True)

            image_data[6]=imageeditor.mostSimilarImage(images)

        image_data[7]=(imageeditor.getAverageHistogram(images))

        image_data[8]=image_data[7]

        similarities = [{},{},{},{},{},{},{},{},{}]

        for country in countries:

            similarities[0][country] =  1 - imageeditor.compareImagesNormalizedRootMeanSquareError(image_data[0],country_data[0][country])
            similarities[1][country] = 1 - imageeditor.compareImagesNormalizedRootMeanSquareError(image_data[1], country_data[1][country])

            if not panoramas:

                similarities[2][country] = 1 - imageeditor.compareImageCluster(image_data[2],country_data[2][country])
                similarities[3][country] = 1 - imageeditor.compareImageCluster(image_data[3], country_data[3][country])

                similarities[4][country] = 1 - imageeditor.compareImageCluster(image_data[4], country_data[4][country],weighted=False)
                similarities[5][country] = 1 - imageeditor.compareImageCluster(image_data[5], country_data[5][country],weighted=False)

                similarities[6][country] = 1 - imageeditor.compareImagesNormalizedRootMeanSquareError(image_data[6], country_data[6][country])

            similarities[7][country] = 1 - imageeditor.compareHistogram(image_data[7],country_data[7][country])
            similarities[8][country] = 1 - imageeditor.compareHistogram(image_data[8], country_data[8][country],steps=5)

        for index,similarity in enumerate (similarities):
            similarities[index] = dict(sorted(similarity.items(), key=lambda item: item[1],reverse=True))

        print (similarities)

        for index,placing in enumerate (placings):
            place = getCountryPlacing(image_country,similarities[index].items())
            if place:
                placings[index] = placing + place
            else:
                placings[index] = 0

    for index, placing in enumerate(placings):
        placings[index] = placing / image_number

    print (placings)


#test both captioning models of the context detection module
def testCaptionModels(countries=None,number = 10):

    if not countries:
        countries = ['KHM', 'HKG', 'BEL', 'GUM', 'IRL', 'GHA', 'BRA', 'CHL', 'KGZ', 'BOL', 'HUN', 'GBR', 'DOM', 'FIN', 'CUW', 'PCN', 'PRT', 'TUN', 'ALB', 'SGP']

    placings = [0,0,0,0,0,0]
    similarities = [0,0,0,0,0,0]

    for index in range (number):

        country = random.choice(countries)

        coordinates = False
        while not coordinates:
            coordinates = datamanager.getPointInCountry(country=datamanager.getCountryNumber(country))

        images = datamanager.getPanoramaAtPitch(coordinates,pitch=0)

        similarities.append(contextdetector.analyseWordLists(images,number=100,model="coco",mode=1))
        similarities.append(contextdetector.analyseWordLists(images,number=100,model="conceptual",mode=1))

        similarities.append(contextdetector.analyseWordLists(images,number=200,model="coco",mode=1))
        similarities.append(contextdetector.analyseWordLists(images,number=200,model="conceptual",mode=1))

        similarities.append(contextdetector.analyseWordLists(images,number=300,model="coco",mode=1))
        similarities.append(contextdetector.analyseWordLists(images,number=300,model="conceptual",mode=1))

        for index,placing in enumerate (placings):
            placings[index] = placings[index] + getCountryPlacing(country=country,result=similarities[index])

    for placing in placings:
        print (placing)

#test the license plate detection
def testLicensePlateDetection(countries = None,number = 40,modes=[1,1],prescan = True, panoramas = False):

    right_guesses = []
    wrong_guesses = []

    not_guessed = 0

    if countries == None:
        countries = datamanager.getAllCountriesWithExternalData()

    for index in range(number):

        car = False

        while not car:

            country = random.choice(countries)

            coordinates = False

            while not coordinates:
                coordinates = datamanager.getPointInCountry(datamanager.getCountryNumber(country))

            if not panoramas:
                images = [datamanager.getImageWithDict(coordinates)]
            else:
                images = datamanager.getPanoramaAtPitch(coordinates, pitch=0)

            for image in images:
                objects = objectdetector.detectObjects(image)
                print (objects)

                for object in objects:
                    if object[6] == "car" or not prescan:
                        car = True

            guessed_countries = []

            if car:
                if modes[0]:
                    guessed_countries = objectdetector.analyseObjects(images,mode=0)
                if modes[1]:
                    guessed_countries = objectdetector.analyseObjects(images,mode=1)

                if len (guessed_countries) == 0:
                    not_guessed = not_guessed + 1

                for guessed_country in guessed_countries:
                    print (country, guessed_country[0])
                    if country == guessed_country[0]:
                        right_guesses.append(guessed_country)
                    else:
                        wrong_guesses.append(guessed_country)

    average_confidence = 0
    other_confidence = 0

    for country in right_guesses:
        average_confidence = average_confidence + country[1]

    for country in wrong_guesses:
        other_confidence = other_confidence + country[1]

    if len (right_guesses) > 0:
        average_confidence = average_confidence / len (right_guesses)
    else:
        average_confidence = 0

    if len (wrong_guesses) > 0:
        other_confidence = other_confidence / len(wrong_guesses)
    else:
        other_confidence = 0

    print ("right: ", len (right_guesses))
    print("percentage: ", len(right_guesses) / number )

    print ("no guess possible: ",not_guessed)
    print ("percentage of actual guesses", len(right_guesses) / (number - not_guessed) )

    print ("average confidence: ", average_confidence)

    print ("wrong", len (wrong_guesses))

    print("other confidence: ", other_confidence)

#test the general object detection
def testObjectLists(countries=None,image_number=100,list_numbers=[100],fov=90):

    results_normal = [0] * len(list_numbers)
    results_weighted = [0] * len(list_numbers)

    densities = [0] * len(list_numbers)
    densities_difference = [0] * len(list_numbers)

    results_combined = [0] * len(list_numbers)

    guess_possible = 0

    if countries == None:
        countries = datamanager.getAllCountriesWithExternalData()


    for panorama in range(image_number):

        country = random.choice(countries)

        coordinates = False

        while not coordinates:
            coordinates = datamanager.getPointInCountry(datamanager.getCountryNumber(country))

        images = datamanager.getPanoramaAtPitch(coordinates, pitch=0)

        for index,number in enumerate (list_numbers):

            result_normal = objectdetector.analyseCountryObjects(images,number,fov,False)
            result_weighted = objectdetector.analyseCountryObjects(images,number,fov,True)

            density = objectdetector.analyseDensity(images,number,fov,mode=0)
            density_difference =  objectdetector.analyseDensity(images,number,fov,mode=1)

            densities[index] = densities[index] + getCountryPlacing(country, density)
            densities_difference[index] = densities_difference[index] + getCountryPlacing(country, density_difference)

            if result_normal:

                results_normal[index] = results_normal[index] + getCountryPlacing(country,result_normal)
                results_weighted[index] = results_weighted[index] + getCountryPlacing(country,result_weighted)

                results_combined[index] = results_combined[index] + ( getCountryPlacing(country,result_weighted) + getCountryPlacing(country, density) ) / 2

                if index == 0:
                    guess_possible = guess_possible + 1

    for index,result in enumerate (results_normal):
        print (index,result)
        results_normal[index] = result / guess_possible

    for index,result in enumerate (results_weighted):
        print (index,result)
        results_weighted[index] = result / guess_possible

    for index,result in enumerate (densities):
        print (index,result)
        densities[index] = result / image_number

    for index,result in enumerate (densities_difference):
        print (index,result)
        densities_difference[index] = result / image_number

    for index,result in enumerate (results_combined):
        print (index,result)
        results_combined[index] = result / guess_possible

    print ("possible guesses :", guess_possible)

    print ("results normal: ",results_normal)
    print("results weighted: ",results_weighted)

    print ("density normal: ",densities)
    print("density difference: ",densities_difference)

    print ("combinded density: ",results_combined)


