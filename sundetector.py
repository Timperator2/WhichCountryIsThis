import random

import datamanager
import imageeditor
import io
from PIL import Image,ImageOps

#find the number of pixels brighter than a certain threshold
def findBrightPixels(image,thresh):
    img = Image.open(io.BytesIO(image))
    #datamanager.saveTempImage(img)
    #img.show()
    img = ImageOps.grayscale(img) #make image grayscale to only include brightness
    width, height = img.size
    pixels_above = 0
    total_brightness = 0
    #datamanager.saveTempImage(img)
    #img.show()
    for w in range(width): #iterate over all pixels
        for h in range(height):
            brightness = img.getpixel((w, h))
            if brightness < thresh: #when the brightness of a pixel is below a certain threshold
                img.putpixel((w, h), 0) #filter that pixel out by making it black
            else:
                total_brightness = total_brightness + brightness #increase total brightness value
                pixels_above=pixels_above+1 #one pixel more above threshold

    #datamanager.saveTempImage(img)
    #img.show()
    return img,total_brightness / (pixels_above) #return ratio of total brightness and pixels above

#find the brightest images
def findSun(images,thresh=100):
    brightness = [ -1.0,-1.0 ]
    brightest = []
    for image in images: #for each image
        #print ("searching for sun in image with heading",image["heading"])
        values = findBrightPixels(image["file"],thresh) #get the brightness score
        brightness[0]=values[1]
        brightness[1]=image["heading"]
        brightest.append(brightness.copy()) #add the score and the heading to a list

    # sort the list so that the heading with the highest score is at index 0
    brightest.sort(reverse=True)

    return brightest

#gets a preciser sun position with consideration fo neighboring images
def getPreciserSunPos(values):
    if (isNextSunNext(values)): #when the second brightest image is a neighbor
        return (float(values[0][1]) * 2 + float(values[1][1])) / 3 #slighlty shift position to the neighbor
    else:
        return 90 #could be any position where the sun position is not clear


#check if second brightest image is a neighbor
def isNextSunNext(values):
    headings = [ int(x[1]) for x in values ]
    headings.sort()
    fov = headings[1] - headings[0] #calculate fov by getting the difference in heading between two neighbors
    if  abs (int (values[0][1]) - int (values[1][1])) == fov: #check if the heading distance of the brightest and the second brightest image matches fov
        return True
    else:
        return False

#get hemishphere for the closest celestial direction to sun position
def getHemissphere(sun_dir):
    closest =  min( [0,90,180,270,360], key=lambda x: abs(x - sun_dir)) #get clostest celestial direction
    if (closest == 0) or (closest == 360):
        print ("southern hemisphere")
        return "south"
    elif closest == 180:
        print ("northern hemishphere")
        return "north"
    else:
        print ("unclear hemisphere")
        return "unclear"

#increase likelihood for each country in the given hemisphere
def countriesForHemissphere(hemi):

    if hemi == "unclear":
        return []

    countries = datamanager.getAllCountriesWithExternalData()
    results = []
    for country in countries:
        facts = datamanager.loadCountryFacts(country)
        if facts["hemisphere"] == hemi: #when country is on the fitting hemisphere
            results.append([country,1.0]) #increase its likelihood

    return results

#analyze celestial direction of the sun and increase likelihood for countries in the fitting hemisphere
def analyseSun(images,zoom_fov=0,mode=0):
    print ("searching for sun")
    brightest = findSun(images) #find heading where the image is brightest
    #print (brightest)
    preciser_pos = getPreciserSunPos(brightest) #get a preciser position
    hemi = countriesForHemissphere(getHemissphere(preciser_pos)) #get all countries in the hemisphere with increased likelihood
    if (zoom_fov == 0):
        return hemi
    elif (hemi == "unclear" and isNextSunNext(brightest)): #when second brightest image is neighbor but the hemisphere is still unclear
        print ("zooming for better sun detection")
        return zoomSun(images,brightest,zoom_fov,mode) #zoom and try again
    else:
        return hemi

#get zoomed images for the headings of the brightest and the second brightest image
def zoomSun(images,values,new_fov,mode=0,debug=False):
    brightest = values[0][1]
    second_brightest = values[1][1]
    if mode==0: #zoom by making another API call
        coordinates = images[0]["coordinates"]
        pitch = images[0]["pitch"]
        images = datamanager.getPanoramaAtPitch(coordinates,pitch=pitch,fov=new_fov,heading_start= min(brightest,second_brightest),heading_end= max(brightest,second_brightest),debug=debug)
        datamanager.saveTestImages(images)
    else: #zoom by cropping the existing images
        new_images = []
        for image in images:
            if (image["heading"] == brightest or image["heading"] == second_brightest):
                new_images.append(image)
        images = imageeditor.zoomedImagesFromExisting(new_images, new_fov)

    return analyseSun(images) #try analyzing again

