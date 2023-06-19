import shapely.geometry
import yolov5
import textandlanguagedetector
import datamanager
from PIL import Image
import imageeditor
import io
import numpy
import cv2
import contextdetector
import math
from shapely import geometry, affinity, ops

model = None

#load the yolo model and set parameters
def loadModel():

    print ("loading yolov5")

    global model

    model = yolov5.load("yolov5s.pt")
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

#detect objects in an image
def detectObjects(image):

    img = Image.open(io.BytesIO(image["file"]))

    results = model(img, augment=True) # inference with test time augmentation

    #results.save(save_dir="C:/Users/Tim/GetImageTestRequest/temporary_images/")
    #results.show()

    return results.pandas().xyxy[0].values.tolist() #return detected objects

#crop out all cars from the image
def cropCars(image,objects):

    cars = []

    for object in objects:
        if object[6] == "car": #if the object is classified as car
            cropped = imageeditor.cropObject(image,(object[0],object[1],object[2],object[3]))
            if cropped.width * cropped.height > 5000: #if the size of the cutout is high enough
                cars.append(cropped)

    return cars

#locate the position of objects to zoom in
def locateObjects(objects,x=640,y=640):

    positions = [False, False, False, False, False, False, False, False,False]  # upleft.midleft,downleft,upmid and so on
    rectangles = []
    width_step = x / 3
    height_step = y / 3

    for w in range(3): #split the image in nine squares
        current_width = w * width_step
        for h in range(3):
            current_height = h * height_step
            rectangles.append(geometry.box(current_width, current_height, current_width + width_step, current_height + height_step))

    for bounding in objects:
        if bounding[6]=="car": #get the boundig box of the detected car
            bounding = geometry.box(bounding[0],bounding[1],bounding[2],bounding[3])

            for index, rectangle in enumerate(rectangles):
                if (rectangle.intersects(bounding)): #if the bounding box is in one square
                    positions[index] = True

    return positions

#zoom in by using the zoom method from the textandlangaugedetector
def zoomObject(image,positions):

    return textandlanguagedetector.zoomText(image,positions,20,pitch_scale=4)

#detect the license plate of a car
def detectLicensePlate(image):

    blurred_areas = numpy.asarray(imageeditor.highlightBlur(image)) #highlight all blurred areas

    ret, thresh = cv2.threshold(blurred_areas, 127, 255, 0) #threshold the image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #find contours

    if not contours:
        return False

    best_poly = None
    max_score = 0
    ratio = 0

    typical_point = (image.width/2, image.height - image.height/3) #a typical license plate location

    for cnt in contours:
        area = cv2.contourArea(cnt) #get the area of all contours

        if area > 0: #if contour has an area (encloses space)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = numpy.int0(box)
            temp_poly = geometry.Polygon([[p[0], p[1]] for p in box])
            middle_point = (temp_poly.centroid.x,temp_poly.centroid.y) #get the middle point of the contour
            poly_width = abs (temp_poly.exterior.bounds[0] - temp_poly.exterior.bounds[2])
            poly_height =  abs (temp_poly.exterior.bounds[1] - temp_poly.exterior.bounds[3])
            temp_ratio = min (poly_width / max (poly_height,0.01),2.0) #get the aspect ratio of the contour (maxed out at two)
            position_score = (area - math.dist(typical_point,middle_point)) * temp_ratio #calculate a confidence score

            #print ("score",position_score)

            if position_score > max_score:
                max_score = position_score
                best_poly = temp_poly #get the shape with the highest confidence value
                ratio = temp_ratio

    if best_poly and ratio > 0.75:
        best_poly = affinity.scale(best_poly, xfact=1.5, yfact=1.5, zfact=1.5, origin='center')
    else:
        return False

    coords = list(best_poly.exterior.coords[:-1])

    imageeditor.drawContours(image.copy(),coords)

    return best_poly.bounds

#detect the license plate by checking for low variances of the laplacian
def detectLicensePlateLaplacian(image):

    og_image = image.copy()

    width, height = image.size
    image = image.convert("L") #convert to grayscale

    tiles = imageeditor.tileImageNumpy(image,int (width/8),int (height/8)) #split the image into several tiles 6x6

    variances = []

    for tile in tiles:
        variances.append([cv2.Laplacian(tile, cv2.CV_64F).var(),tile]) #calculate the variance of the laplacian

    variances.sort()

    larger_templates = [template for template in variances if template[1].shape[::-1][0] * template[1].shape[::-1][1] > (width * height) / 150] #/100
    lowest_variances = larger_templates[0:int (len(larger_templates) * 0.2)] #get the twenty percent of tiles above a certain size with the lowest variances

    image = numpy.asarray(image)

    rectangles = []

    for template in lowest_variances:

        w, h = template[1].shape[::-1]
        res = cv2.matchTemplate(image, template[1], cv2.TM_CCOEFF_NORMED) #search for tile in image

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) #get min and max values for coordinates
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        rectangle = shapely.geometry.box(top_left[0],bottom_right[1],bottom_right[0],top_left[1]) #create rectangle for it
        rectangles.append(rectangle)

    for index,poly in enumerate (rectangles): #rescale all rectangles for merging
        rectangles[index] = affinity.scale(poly, xfact=1.5, yfact=0.75, zfact=1.0, origin='center')

    rectangles = shapely.ops.unary_union(rectangles) #merge all intersecting rectangles

    typical_point = (width / 2, height - height / 3) #get a typical point for the license plate location

    max_score = 0
    best_shape = False

    if rectangles.type != "MultiPolygon":
        rectangles = [rectangles]

    for shape in rectangles: #calculate confidence score based on horizontal length and distance to typical point
        score = shape.length - math.dist(typical_point,(shape.centroid.x,shape.centroid.y))
        if score > max_score:
            max_score = score
            best_shape = shape #get the shape with the highest confidence value

    best_shape = affinity.scale(best_shape, xfact=0.5, yfact=1.25, zfact=1.0, origin='center') #rescale shape

    imageeditor.drawContours(og_image,list(best_shape.exterior.coords[:-1]))

    return best_shape.bounds

#crop out the license plate with a given bounding box
def cropLicensePlate(image,license_plate):

    if license_plate:
        return imageeditor.cropObject(image,license_plate)
    else:
        return False

#get the area and aspect ratio of a bounding box
def getBoxSize(box):

    aspect_ratio = abs(box[2] - box[0]) / abs(box[3] - box[1])
    box = geometry.box(box[0],box[1],box[2],box[3])

    return [box.area,aspect_ratio]

#check for license plate colors
def checkLicensePlate(image):

    result = []

    if image:  #check for colors and their location

        #datamanager.saveTempImage(image)

        blue = imageeditor.checkForColorAndSides(image,(120,185),0.08)
        yellow = imageeditor.checkForColorAndSides(image, (28,50), 0.04)
        lower_red = imageeditor.checkForColorAndSides(image, (0, 28),0)
        higher_red = imageeditor.checkForColorAndSides(image, (235, 255),0)
        red = [max (lower_red[0] + higher_red[0] - 0.05,0),(lower_red[1]+higher_red[1]),(lower_red[2]+higher_red[2]),(lower_red[3]+higher_red[3])]

        if blue[0] > 0: #when blue is present
            if blue[1] >= 0.35 and blue[3] >= 0.35: #left and right
                result.append("double blue")
            else:
                result.append("blue")

        if yellow[0] > 0: #when yellow is present
            if yellow[1] > 0.6 or yellow[3] > 0.6: #on any side
                result.append("side yellow")
            else:
                result.append("full yellow")

        if red[0] > 0: #if red is present
            if red[1] > 0.6 or red[3] > 0.6: #on any side
                result.append("side red")
            else:
                result.append("full red")

        if red[0] > yellow[0]: #remove yellow when there is more red
            if "side yellow" in result:
                result.remove("side yellow")
            elif "full yellow" in result:
                result.remove("full yellow")
        elif yellow[0] > red[0]: #remove red when there is more yellow
            if "side red" in result:
                result.remove("side red")
            elif "full red" in result:
                result.remove("full red")

        return result


#search all fact sheets for countries whose license plates fit the color pattern
def getCountryByLicensePlate(plate_colors):

    countries = datamanager.getAllCountriesWithExternalData()

    results = []

    for country in countries:
        facts = datamanager.loadCountryFacts(country)
        for plate in plate_colors:
            if plate:
                for color in plate:
                    #print(color, facts["license colors"])
                    if color and color in facts["license colors"]: #if color pattern in fact sheet
                        results.append([country,1/len(plate_colors)]) #set likelihood based on number of color patterns

    for index,res in enumerate (results): #remove duplicate country mentions

        remaining_results = results.copy()
        remaining_results.remove(res)

        for country in remaining_results:
            if res[0] == country[0]:
                results[index][1] = results[index][1] + country[1]
                results.remove(country)

    return results


loadModel()

#check for visible license plate colors and assign country likelihoods accordingly
def analyseObjects(images,zoom_allowed=True,mode=1):

    print ("searching for license plates")

    objects_to_zoom = []
    results = []

    for image in images:
        objects = detectObjects(image) #detect all objects in the image
        for object in objects:
            #print (getBoxSize(object))
            if object[6] == "car" and getBoxSize(object)[0] < 7500 and zoom_allowed:
                objects_to_zoom.append(object) #mark detected object for zoom
                objects.remove(object)

        zoom_positions = locateObjects(objects_to_zoom) #get image areas to zoom in
        #print (zoom_positions)
        zoomed = zoomObject(image,zoom_positions) #zoom in

        for zoomed_image in zoomed:
            zoomed_objects = detectObjects(zoomed_image)
            zoomed_cars = cropCars(zoomed_image,zoomed_objects)
            for zoomed_car in zoomed_cars:
                if mode == 0: #variane of the laplacian
                    zoomed_plate = cropLicensePlate(zoomed_car, detectLicensePlateLaplacian(zoomed_car))
                elif mode == 1: #low pass filter
                    zoomed_plate = cropLicensePlate(zoomed_car,detectLicensePlate(zoomed_car))

                results.append(checkLicensePlate(zoomed_plate))

        cars = cropCars(image,objects)
        for car in cars:
            if mode == 0: #variane of the laplacian
                plate = cropLicensePlate(car,detectLicensePlateLaplacian(car))
            elif mode == 1: #low pass filter
                plate = cropLicensePlate(car,detectLicensePlate(car))

            results.append(checkLicensePlate(plate)) #find color pattern of license plate

    print ("found colors: ", results)

    countries = getCountryByLicensePlate(results) #get countries with detected color pattern
    print ("possible countries for color combinations: ", countries)

    return countries


#generate the average list of objects for a number of images
def generateAverageObjectList(images,densities=True,all_guesses=True):

    all_objects = {}
    density = 0

    guess_possible = 0

    for image in images:
        objects = detectObjects(image) #detect the objects with YOLO
        if len (objects) > 0:
            guess_possible = guess_possible + 1
        density = density + len(objects)
        for object in objects:
            if object[6] not in all_objects: #if the object type is not in the list
                all_objects[object[6]] = 1 / len(images)
            else:
                all_objects[object[6]] = all_objects[object[6]] + 1 / len(images)

    if densities and density:
        if all_guesses:
            all_objects["density"] = density / len(images) #calculate the average number of objects
        else:
            if guess_possible > 0:
                all_objects["density"] = density / guess_possible #calculate the average number of objects
            else:
                all_objects["density"] = 0

    return all_objects


#merge object lists of different countries
def getCompleteList(number=100,fov=90):

    countries = datamanager.getAllCountriesWithExternalData()
    all_object_lists = []
    total_objects = {}

    for country in countries: #get all average object lists
        all_object_lists.append(datamanager.loadAverageObjectList(country,number,fov))

    for object_list in all_object_lists: #add up all values
        if object_list:
            for object in object_list:
                if object not in total_objects:
                    total_objects[object] = object_list[object]
                else:
                    total_objects[object] = total_objects[object] + object_list[object]

    return total_objects

#get the relative list for an average object list and the complete list
def getRelativeList(country,number,fov,complete_list):
    country_list = datamanager.loadAverageObjectList(country,number,fov) #load the average list
    if country_list:
        for object in country_list: #calculate share
            country_list[object] = country_list[object] / complete_list[object]

    return country_list

#compare two object lists with the same method used for wordslists
def compareList(list,test_list,weighted=True):
    if weighted: #take into account the number of same objects
        new_list = []
        for object in list:
            for index in range( int (list[object]) ): #for the number of objects of same type
                new_list.append(object) #append the object to list
        list = new_list

    return contextdetector.compareWordLists(list,test_list)



#compare the object list of a number of images with the country lists
def analyseCountryObjects(images,number=100,fov=90,weighted=True):

    print("analysing country objects")

    similarity = {}

    for image in images:

        object_list = generateAverageObjectList([image]) #generate the average object list

        print("found objects: ", object_list)

        if len (object_list) > 0: #if objects were detected

            complete_list = getCompleteList(number,fov)  #get the complete object list

            countries = datamanager.getAllCountriesWithExternalData()

            complete_lists_relative = []

            for country in countries: #get relative list for all countries
                complete_lists_relative.append([country,getRelativeList(country,number,fov,complete_list)])

            for country in complete_lists_relative: #get similarity for all countries
                if country[1]:
                    if country[0] not in similarity:
                        similarity[country[0]] = compareList(object_list, country[1],weighted)
                    else:
                        similarity[country[0]] = similarity[country[0]] + compareList(object_list, country[1],weighted)
    result = []

    for country in similarity.items(): #get average similarity per image
        result.append([country[0],country[1]/len(images)])

    result.sort(key=lambda x: x[1], reverse=True)

    return result

#compare the average number of objects of a set of images with the one of countries
def analyseDensity(images,number=100,fov=90,mode=0):

    print ("analysing object density")

    similarities = []

    object_list = generateAverageObjectList(images,densities=True) #generate the average object list with density

    print ("found objects: ", object_list)

    if len(object_list) > 0: #if there wrere any objects detected in the panorama
        density = object_list["density"]
    else:
        density = 0.01

    countries = datamanager.getAllCountriesWithExternalData()

    for country in countries:
        object_list = datamanager.loadAverageObjectList(country,number,fov,density=True) #load object list with density
        if object_list:
            country_density = object_list["density"] #get country density
            if mode == 0:
                if (density < country_density): #always divide the larger value by the smaller
                    similarities.append([country,density/country_density])
                else:
                    similarities.append([country,country_density/density])

            elif mode == 1:
                similarities.append([country, abs (country_density - density)])

    similarities.sort(key=lambda x: x[1], reverse=True)

    #print ("countries for density: ", similarities)

    return similarities

def analyseGeneralObjectDetection(images,number=100,fov=90,mode=0):
    if mode == 0:
        return analyseCountryObjects(images,number,fov)
    if mode == 1:
        return analyseDensity(images,number,fov)


