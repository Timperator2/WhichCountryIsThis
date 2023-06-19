import copy
import random
import PySimpleGUI as sg

dropdown_mode = ["Street View API","Local Images"]
dropdown_random = ["Completly Random","Equal Country Distribution","Single Country"]
dropdown_number = ["1","2","3","4","5","6","7","8","9","10"]
graph_image_mode = True
switch_ids = False

sky_images = None
horizon_one_images = None
horizon_two_images = None

sky_ids = None
horizon_one_ids = None
horizon_two_ids = None

image_windows = []

zoom_level = 90
last_click = None
selected_country=False
status = "Street View not Available for "
guessed_countries=[]

local_game = False
api_zoom_allowed = True
mask = []

offset_x = 0
offset_y = 0

drag_start = False

total_guesses = 0
rounds_left = 0

player_score = 0
ai_score = 0

searched_country = ""

enclaves = ["LSO","SMR","VAT"]

#redirect print to window text window
sg.Print(do_not_reroute_stdout=False)
print = sg.Print


#default window layout
def getDefaultlayout():

    if graph_image_mode:  # display image elements using a graph
        layout_images = [[sg.Graph(canvas_size=(900, 75), graph_bottom_left=(0, 75), graph_top_right=(900, 0),
                                   enable_events=True, key="sky_graph")],
                         [sg.Graph(canvas_size=(900, 250), graph_bottom_left=(0, 250), graph_top_right=(900, 0),
                                   enable_events=True, key="horizon_one_graph")]]
    else:
        layout_images = sg.Column(layout=[[sg.Image(key="sky")], [sg.Image(key="horizon")]])

    layout = [[sg.Text("Path"), sg.Input(key="file_path"),
               sg.FolderBrowse("Select Folder", key="select_folder", target="file_path"),
               sg.FileBrowse("Select Single Image ", key="select_image", target="file_path"),
               sg.Button("Load as Sky", key="load_sky"), sg.Button("Load as Horizon", key="load_horizon"),
               sg.Checkbox("Use Local Images for Game", default=False, key="local_game")],
              [[sg.Text("Enter Coordinates"), sg.Button("Random Coordinates", key="random_coordinates"),
                sg.Combo(dropdown_random, default_value=dropdown_random[1], key="random_mode"), sg.Text("Country"),
                sg.Input(key="random_country")]],
              [sg.Text("Latitute", key="lat_text"), sg.Input(key="lat"), sg.Text("Longitute", key="long_text"),
               sg.Input(key="long"), sg.Button("Fetch Images", key="fetch_images")],
              [sg.Checkbox("Sun", default=True, key="sun"), sg.Checkbox("Language", default=True, key="lang"),
               sg.Checkbox("Color", default=True, key="color"), sg.Checkbox("Context", default=True, key="context"),
               sg.Checkbox("Plates", default=True, key="plates"), sg.Checkbox("Density", default=True, key="density")],
              [sg.Button("Let the AI make a Guess!", key="ai_guess"),
               sg.Checkbox("Allow API Zoom", default=True, key="allow_zoom"), sg.Button("Play vs AI!", key="play"),
               sg.Text("Number of Guesses:"),
               sg.Combo(dropdown_number, default_value=dropdown_number[4], key="total_guesses"),
               sg.Text("Number of Rounds:"), sg.Input(default_text="10", key="rounds")],
              [sg.Multiline(size=(20, 20), disabled=True, echo_stdout_stderr=True, reroute_stdout=True, autoscroll=True,
                            key="output"), sg.Column(layout=layout_images)],
              [sg.Text("Number of Placings to show:"),
               sg.Combo(dropdown_number, default_value=dropdown_number[2], key="number"), sg.Text("Guessed Country:"),
               sg.Multiline(horizontal_scroll=True, disabled=True, key="result")],
              [sg.Text("Searched Country Placing"), sg.Text(key="placing")]]

    return layout

#switch to play mode
def enterPlayMode(values):

    global total_guesses,rounds_left,window,api_zoom_allowed,local_game

    window.close() #close the default window

    layout_images = [ [sg.Graph(canvas_size=(820,100),graph_bottom_left=(0,100),graph_top_right=(820,0),enable_events=True,key="sky_graph")],
                      [sg.Graph(canvas_size=(820,220),graph_bottom_left=(0,220),graph_top_right=(820,0),enable_events=True,key="horizon_one_graph")],
                      [sg.Graph(canvas_size=(820,220),graph_bottom_left=(0,220),graph_top_right=(820,0),enable_events=True,key="horizon_two_graph")]]

    layout_information= [[sg.Button("Select Country on Map",key="show_map")],
                        [sg.Graph(canvas_size=(150,100),graph_bottom_left=(0,100),graph_top_right=(100,0),background_color="green",key="directions")],
                        [sg.Multiline(size=(20, 20), disabled=True, echo_stdout_stderr=True, reroute_stdout=True,autoscroll=True, key="output")]]

    layout_image_viewer = [[sg.Column(layout=layout_information),sg.Column(layout=layout_images)]]

    window = sg.Window("Country Guesser",layout_image_viewer,resizable=True,finalize=True) #open default window in image viewer mode

    #draw compass
    window["directions"].draw_line((30,30),(90,30),"red",7)
    window["directions"].draw_polygon([(90, 20), (90, 40), (100, 30)], "red", "red")
    window["directions"].draw_text("north",(15, 10))
    window["directions"].draw_text("heading",(50,10))
    window["directions"].draw_text("south",(85, 10))
    window["directions"].draw_line((30, 30), (30, 90), "white", 7)
    window["directions"].draw_polygon([(20,90),(40,90),(30,100)],"white","white")
    window["directions"].draw_text("pitch", (15, 50))

    total_guesses = int (values["total_guesses"])
    rounds_left = int(values["rounds"])
    api_zoom_allowed = values["allow_zoom"]
    local_game = values["local_game"]

    getMask(values) #the modules to be used

    fetchPlayImages() #get images for first round

    openGameWindow() #open the game window

    #window.bring_to_front()

#get images for play mode
def fetchPlayImages():

    global searched_country,sky_ids, horizon_one_ids, horizon_two_ids, sky_images, horizon_one_images, horizon_two_images, switch_ids

    if local_game: #play without API
        file_layout = [[sg.Text("Path"),sg.Input(key="file_path"),sg.FolderBrowse("Select Folder",key="select_folder",target="file_path"),sg.FileBrowse("Select Single Image ",key="select_image",target="file_path"),sg.Button("Load as Sky",key="load_sky"),sg.Button("Load as Horizon",key="load_horizon")],
                       [sg.Text("Searched Country:"),sg.Input(key="searched_country"),sg.Button("Start Round",key="start_round")]]
        file_window = sg.Window("File Picker",file_layout,resizable=True,finalize=True)
        while True:
            event,values = file_window.read()
            if event == "load_sky":
                sky = datamanager.loadGeneralImages(values["file_path"])
                sky_images = sky
                sky_ids = drawImageGraph("sky_graph", sky)
                updateLocalSearchedCountry(sky,file_window)

            if event == "load_horizon":
                horizon_one = datamanager.loadGeneralImages(values["file_path"])
                horizon_one_images = horizon_one
                horizon_one_ids = drawImageGraph("horizon_one_graph", horizon_one)
                updateLocalSearchedCountry(horizon_one,file_window)

            if event == "start_round":
                searched_country = values["searched_country"]
                if not sky_images:
                    mask[0] = 0
                file_window.close()
                break


    else: #play with API
        switch_ids = False
        fovs = [45, 90]
        if not mask[1]: #when playing without text and langauge detector
            fovs = [90]

        searched_country = random.choice(countries)
        country_number = datamanager.getCountryNumber(searched_country)
        coordinates = datamanager.getPointInCountry(country_number)
        searched_country = datamanager.getCountryName(country_number)[0]
        sky,horizon_one,horizon_two = countryguesser.collectImages(coordinates,fovs,debug=False)

        sky_images = sky
        horizon_one_images = horizon_one
        horizon_two_images = horizon_two
        sky_ids = drawImageGraph("sky_graph",sky)

        horizon_one_ids = drawImageGraph("horizon_one_graph", horizon_one)

        if horizon_two:
            #switch_ids = False
            horizon_two_ids = drawImageGraph("horizon_two_graph",horizon_two)

#update searched country based on local images
def updateLocalSearchedCountry(images,window):
    coordinates = images[0]["coordinates"]
    if coordinates:
        coordinates = (float(coordinates[0]), float(coordinates[1]))
        window["searched_country"].update(datamanager.getCountryofPoint(coordinates)[0])

#show ai results
def updateRoundWindow(score,placing,ai_guesses):
    ai_result = "The AI guessed the following countries: " + (",".join(ai_guesses))
    if score == 0:
        ai_result = ai_result + "\nThe AI didnt guess the right country (searched country was guessed at position " + str(placing) + ")"
    else:
        ai_result = ai_result + "\nThe AI guessed the right country at position " + str(placing) +  " and got " + str(score) + " points"

    print (ai_result)

    if round_window:
        round_window["ai_result"].update(ai_result)
        round_window.read(timeout=100)

#let the ai play
def aiRound():

    global ai_score

    country = datamanager.getCountryName(datamanager.getCountryNumber(searched_country))[1]
    results = countryguesser.makeGuess(sky_images,horizon_one_images,horizon_two_images,mask=mask,zoom_allowed=api_zoom_allowed,country=country)
    placing =  countryguesser.getCountryPlacing(country, results)
    results = results[0:total_guesses]
    points_gained = 0
    for index,country in enumerate(results):
        if datamanager.getCountryName(datamanager.getCountryNumber(country[0]))[0] == searched_country:
            points_gained = calculateScore(index)
            ai_score = ai_score + points_gained
            game_window["ai_score"].update(ai_score)

    shortened_result =  [datamanager.getCountryName(datamanager.getCountryNumber(res[0]))[0] for res in results]

    updateRoundWindow(points_gained,placing,shortened_result)


#draw images on graph element
def drawImageGraph(graph,images):

    images = copy.deepcopy(images) #full copy images without any references
    image_ids = []

    window[graph].erase() #clear graph

    if (graph == "sky_graph"):
        for index, image in enumerate(images):
            if image["heading"] == 360:
                images[index]["heading"] = 0 #360 and 0 are the same heading

    canvas_width, canvas_height = window[graph].CanvasSize
    headings = []
    pitches = []
    for index,image in enumerate (images): #collect all headings and pitches
        if image["heading"] in headings and image["pitch"] in pitches:
            images[index]["heading"] = images[index]["heading"] + 1
            headings.append(images[index]["heading"])
        if image["heading"] not in headings:
            headings.append(image["heading"])
        if image["pitch"] not in pitches:
            pitches.append(image["pitch"])

    headings.sort()
    pitches.sort(reverse=True)
    width_step = int(canvas_width / len(headings))
    height_step = int(canvas_height / len(pitches))
    images = imageeditor.resize_images(images, width_step, height_step)
    for index,image in enumerate(images): #draw images according to their heading and pitch
        width_index = headings.index(image["heading"])
        height_index = pitches.index(image["pitch"])
        id = window[graph].draw_image(data=datamanager.ImageToBytes(imageeditor.imageToPIL(image)),location=(width_index * width_step, height_index * height_step))
        image_ids.append([index,id])

    return image_ids

#get the just clicked image
def getClickedImage(graph,click):
    figure = window[graph].get_figures_at_location(click)
    if figure: #if an object was clicked
        figure = figure[0] #get the id

    if graph == "sky_graph":
        ids = sky_ids
        images = sky_images
    elif graph == "horizon_one_graph":
        ids = horizon_one_ids
        if switch_ids: #make sure ids match images
            images = horizon_two_images
        else:
            images = horizon_one_images
    else:
        ids = horizon_two_ids
        images = horizon_two_images

    if ids: #get image with the right id
        for id in ids:
            if id[1] == figure:
                return images[id[0]]


#get random coordinates
def selectRandomCoordinates(mode):
    if mode == dropdown_random[0]:#completly random
        coordinates = datamanager.findAvailableRandomImage()
        window["random_country"].update(datamanager.getCountryofPoint(coordinates)[0])
    elif mode == dropdown_random[1]: #equal country distribution
        country = random.choice(countries)
        country_number = datamanager.getCountryNumber(country)
        window["random_country"].update(datamanager.getCountryName(country_number)[0])
        coordinates = datamanager.getPointInCountry(country_number)
    elif mode == dropdown_random[2]: #inside single country
        country = values["random_country"]
        coordinates = datamanager.getPointInCountry(datamanager.getCountryNumber(country))

    window["lat"].update(coordinates[0])
    window["long"].update(coordinates[1])

#load images from local path
def loadImages(path,mode):

    images = datamanager.loadGeneralImages(path)
    coordinates = images[0]["coordinates"]

    if coordinates:
        window["lat"].update(coordinates[0])
        window["long"].update(coordinates[1])

        coordinates = (float(coordinates[0]),float(coordinates[1]))
        window["random_country"].update(datamanager.getCountryofPoint(coordinates)[0])

    if mode == 0: #if loaded as sky
        updateSkyImage(images)

    if mode == 1: #if loaded as horizon
        updateHorizonImage(images)

#form panorama image out of sky images
def getSkyPanorama(sky):

    sky = copy.deepcopy(sky) #completly copy the sky images without any references

    sky_top = [image for image in sky if image["pitch"] == 90] #get sky image facing upwards

    if sky_top:
        sky_top = sky_top[0]
        sky.remove(sky_top) #remove top image

    sky_image = imageeditor.mergePanorama(sky) #merge remaining images to panorama

    if sky_top:
        sky_image = imageeditor.mergeTwoImages(sky_image, sky_top, mode=0) #stitch top image on top of the panorama
        sky_image = datamanager.packImageDict(datamanager.ImageToBytes(sky_image), sky_top["coordinates"], 0, 360, 0)

    return sky_image

#draw sky image
def updateSkyImage(sky):

    global sky_images,sky_ids

    sky_images = sky

    if graph_image_mode:
        sky_ids = drawImageGraph("sky_graph", sky)
    else:
        if len(sky) > 1:
            sky_image = getSkyPanorama(sky)
            sky_image = imageeditor.resize_images([sky_image], 950, 75)[0]
        else:
            sky_image = sky[0]
            sky_image = imageeditor.resize_images([sky_image], 450, 225)[0]

        window["sky"].update(data=sky_image["file"])


#form panorama image out of horizon images
def getHorizonPanoramas(horizon_one,horizon_two=False):

    global horizon_one_images, horizon_two_images

    second_horizon_image = None

    if horizon_two: #if there are two horizon panoramas
        horizon = horizon_two
        second_horizon_image = imageeditor.mergePanorama(horizon_one)
    else:
        horizon = horizon_one

    horizon_image = imageeditor.mergePanorama(horizon)

    return [horizon_image, second_horizon_image]

#draw horizon image
def updateHorizonImage(horizon_one,horizon_two=False):

    global horizon_one_images, horizon_two_images,horizon_one_ids, switch_ids

    if horizon_two: #if there are two horizon panoramas
        horizon_one_images = horizon_one
        horizon_two_images = horizon_two
        horizon = horizon_two
        switch_ids = True #make sure ids match images
    else:
        horizon_one_images = horizon_one
        horizon_two_images = None
        horizon = horizon_one
        switch_ids = False

    if graph_image_mode:
        horizon_one_ids = drawImageGraph("horizon_one_graph", horizon)
    else:
        if len(horizon_one) > 1:
            horizon_image = getHorizonPanoramas(horizon_one,horizon_two)[0]
            horizon_image = imageeditor.resize_images([horizon_image], 950, 250)[0]
        else:
            horizon_image = horizon_one[0]
            horizon_image = imageeditor.resize_images([horizon_image], 450, 225)[0]

        window["horizon"].update(data=horizon_image["file"])


#collect images
def fetchImages(text_and_lang):

    fovs = [45,90]
    if not text_and_lang:
        fovs=[90]

    sky,horizon_one,horizon_two = countryguesser.collectImages((values["lat"], values["long"]),fovs)

    return sky,horizon_one,horizon_two

#collect images and update sky and horizons
def fetchAndUpdate(text_and_lang):

    sky,horizon_one,horizon_two = fetchImages(text_and_lang)

    updateSkyImage(sky)

    updateHorizonImage(horizon_one,horizon_two)

#update settings depending on available images for guessing
def updateSettings(values):
    if sky_images == None:
        print ("No sky images found, adapting mask")
        window["sun"].update(False)
    if horizon_one_images == None:
        print("No horizon images found, adapting mask")
        window["lang"].update(False)
        window["color"].update(False)
        window["context"].update(False)
        window["plates"].update(False)
        window["density"].update(False)

    if values["lat"] == "" or values["long"] == "":
        window["allow_zoom"].update(False)

#let the ai make a guess
def aiGuess(mask):

    country = None

    if values["random_country"] != "":
        country = datamanager.getCountryName(datamanager.getCountryNumber(values["random_country"]))[1]

    zoom_allowed = values["allow_zoom"]
    result = countryguesser.makeGuess(sky_images,horizon_one_images,horizon_two_images,mask=mask,country=country,zoom_allowed=zoom_allowed)
    places = ""
    for i in range(int(values["number"])):
        places = places + " " + datamanager.getCountryName(datamanager.getCountryNumber(result[i][0]))[0] + " " + str(round(result[i][1], 2))

    window["result"].update(places)

    if country:
        placing = countryguesser.getCountryPlacing(datamanager.getCountryName(datamanager.getCountryNumber(values["random_country"]))[1], result)
        print ("overall placing:",placing)
        window["placing"].update(placing)


#check if point is in enclave
def handleEnclaves(point):

    for country in enclaves:
        country_number = datamanager.getCountryNumber(country)
        polygons = datamanager.getCountryPolygons(country_number)
        if datamanager.isPointInCountry(point,polygons): #if clicked coordinate is in enclave
            return datamanager.getCountryName(country_number)

    return False

#reset enclaves color
def handleEnclaveNeighbours(country):
    enclave_neighbours = ["Italy","South Africa"]
    if country in enclave_neighbours:
        colorCountries(enclaves,"white")

#draw map with all country polygons
def drawMap():

    for country in datamanager.getCountryNumberList():
        polygons = datamanager.getCountryPolygons(country)
        for poly in polygons:
            coords = poly.exterior.coords
            game_window["graph"].draw_polygon(coords,fill_color="white",line_color="black")


#get area after zoom
def getZoomArea(coordinates,scale):

    bottom_left = (coordinates[0] - scale * 2,coordinates[1] - scale)
    bottom_right = (coordinates[0] + scale * 2,coordinates[1] + scale )

    return (bottom_left,bottom_right)

#zoom and redraw map accordingly
def zoomMap(mouse,strenght,values,mode=1):

    global game_window,zoom_level,offset_x,offset_y

    offset_x = 0
    offset_y = 0

    if mode == 0: #zoom with opening new window
        old_game_window = game_window #save old map
        old_game_window.keep_on_top_set() #keep in foreground

    zoom_level = zoom_level * strenght #calculate zoom direction

    zoom_area = getZoomArea(mouse, zoom_level) #new area after zoom

    if mode == 1: #zoom in existing window
        game_window["graph"].erase()
        game_window["graph"].change_coordinates(zoom_area[0], zoom_area[1])

    if mode == 0:

        #redraw game window with changed map area

        left_column = [    [sg.Button("Back to Panorama",key="show_images")],
                           [sg.Text("Own Score:")],
                           [sg.Text("", key="own_score")],
                           [sg.Text("Ai Score:")],
                           [sg.Text("", key="ai_score")],
                           [sg.Text("Guesses Left:")],
                           [sg.Text("", key="guesses_left")],
                           [sg.Text("Rounds Left:")],
                           [sg.Text("", key="rounds_left")]]

        zoomed_layout = [[sg.Column(layout=left_column),sg.Graph(canvas_size=(720, 360), graph_bottom_left=zoom_area[0], graph_top_right=zoom_area[1],background_color="LightSkyBlue1", key="graph", float_values=True, drag_submits=True,enable_events=True)],
                         [sg.Text("Country"), sg.Input(key="country_guess"), sg.Button("Guess",key="guess"),sg.Text("Already Guessed:"),sg.Text("",key="already_guessed")]]
        game_window = sg.Window("Select Country", zoomed_layout, location=old_game_window.current_location(),finalize=True, return_keyboard_events=True,resizable=True)

    drawMap() #redraw map after zoom

    if selected_country: #update last selected country
        #print (selected_country,status)
        if "not" in status:
            colorCountry(selected_country,"Gray")
        else:
            colorCountry(selected_country, "Green")

        handleEnclaveNeighbours(selected_country)

    #restore values from old game window

    for country in guessed_countries:
        colorCountry(country,"Red")
        handleEnclaveNeighbours(country)

    if mode == 0:
        game_window["country_guess"].update(values["country_guess"])
        game_window["guesses_left"].update(total_guesses - len(guessed_countries))
        game_window["rounds_left"].update(rounds_left)
        game_window["already_guessed"].update(",".join(guessed_countries))
        game_window["own_score"].update(player_score)
        game_window["ai_score"].update(ai_score)

        old_game_window.close() #close the old game window

#color a country  by placing a colored copy of its polygon on top
def colorCountry(country,color):
    country = datamanager.getCountryPolygons(datamanager.getCountryNumber(country))
    for poly in country:
        coords = poly.exterior.coords[:]
        if offset_x != 0 or offset_y != 0: #if the polygon was moved from the original coordinates
            coords = coords[:]
            for index,c in enumerate (coords):
                coords[index] = (c[0]+offset_x, c[1]+offset_y) #calculate new coordinates with offset
        game_window["graph"].draw_polygon(coords, fill_color=color, line_color="black")

#color a list of countries
def colorCountries(countries,color):
    for country in countries:
        colorCountry(country,color)

#remove country from selection
def deselectCountry():
    global  selected_country
    if selected_country and selected_country not in guessed_countries:
        colorCountry(selected_country, "White")
        selected_country = None

#change selected country
def updateSelectedCountry(new_selected_country):

    global selected_country,status

    if new_selected_country and (new_selected_country[0] not in guessed_countries): #if new country
        deselectCountry() #deselect old selection
        selected_country = new_selected_country[0]
        if datamanager.isStreetViewAvailable(datamanager.getCountryNumber(selected_country), api=False):
            status = ""
            colorCountry(selected_country, "Green")
            game_window["country_guess"].update(selected_country)
        else:
            status = "Street View not Available for "
            colorCountry(selected_country, "Gray")

        game_window["country_guess"].update(status + selected_country)

#open the game window
def openGameWindow():

    global game_window

    left_column = [    [sg.Button("Back to Panorama",key="show_images")],
                       [sg.Text("Own Score:")],
                       [sg.Text("", key="own_score")],
                       [sg.Text("Ai Score:")],
                       [sg.Text("", key="ai_score")],
                       [sg.Text("Guesses Left:")],
                       [sg.Text("", key="guesses_left")],
                       [sg.Text("Rounds Left:")],
                       [sg.Text("", key="rounds_left")]]

    #game window layout
    layout_game = [[sg.Column(layout=left_column),sg.Graph(canvas_size=(720, 360), graph_bottom_left=(-180, -90), graph_top_right=(180, 90),
                             background_color="LightSkyBlue1", key="graph", float_values=True, drag_submits=True,
                             enable_events=True)],
                   [sg.Text("Country"), sg.Input(key="country_guess"), sg.Button("Guess", key="guess"),sg.Text("Already Guessed:"),sg.Text("",key="already_guessed")]]

    #open game window
    game_window = sg.Window("Select Country", layout_game, resizable=True, finalize=True, return_keyboard_events=True)

    game_window.send_to_back()

    #set values
    game_window["guesses_left"].update(total_guesses)
    game_window["rounds_left"].update(rounds_left)
    game_window["own_score"].update(player_score)
    game_window["ai_score"].update(ai_score)

    drawMap() #draw map with country polygons

#get the modules which are to be used
def getMask(values):
    global mask
    mask = [values["sun"], values["lang"], values["color"], values["context"], values["plates"],
            values["density"]]

#calculate the score for a guess
def calculateScore (already_guessed):

    return total_guesses - already_guessed

#update the score of a player
def updatePlayerScore():
    global player_score
    points_gained = calculateScore(len(guessed_countries))
    player_score = player_score + points_gained
    game_window["own_score"].update(player_score)
    return points_gained

#move on to the next guess for a round
def nextGuess():
    global guessed_countries
    if selected_country not in guessed_countries:
        guessed_countries.append(selected_country)
        colorCountry(selected_country, "Red")
        handleEnclaveNeighbours(selected_country)
        game_window["guesses_left"].update(int(total_guesses) - len(guessed_countries))
        game_window["already_guessed"].update(",".join(guessed_countries))

#move on to the next round in a game
def nextRound(score):
    global  rounds_left,guessed_countries, image_windows, game_window

    showRoundResults(score)

    rounds_left = rounds_left - 1

    aiRound()

    if rounds_left > 0:
        game_window["rounds_left"].update(rounds_left)
        colorCountries(guessed_countries,"white")
        guessed_countries = []
        game_window["guesses_left"].update(int(total_guesses))
        game_window["already_guessed"].update(",".join(guessed_countries))
        fetchPlayImages()

    window.bring_to_front() #move the image viewer in front of the game window
    round_window.bring_to_front() #bring results to foreground

    round_window["next_round"].update(visible=True)

    for image_window in image_windows: #close all open old image windows
        image_window.close()
        image_windows.remove(image_window)

    if rounds_left == 0:
        #window.close()
        game_window.close()
        game_window = None
        showGameResults()


#enlarge and show a clicked image
def displayClickedImage(event,values):

    global image_windows

    image = getClickedImage(event, values[event])
    if image:
        image_layout = [[sg.Image(datamanager.ImageToBytes(imageeditor.imageToPIL(image)))]]
        image_name = "heading" + str(image["heading"]) + "fov" + str(image["fov"]) + "pitch" + str(image["pitch"])
        image_windows.append(sg.Window(image_name, image_layout, resizable=True, finalize=True))

#handle a click on the map
def handleMapClick(values):

    global last_click, offset_x, offset_y, drag_start

    if not drag_start: #if not during drag movement
        last_click = list(values["graph"]) #get coordinate of mouse click
        last_click[0] = last_click[0] - offset_x
        last_click[1] = last_click[1] - offset_y
        new_selected_country = handleEnclaves((last_click[1], last_click[0])) #check if enclave
        if not new_selected_country: #if clicked country is not an enclave
            new_selected_country = datamanager.getCountryofPoint((last_click[1], last_click[0]))
        updateSelectedCountry(new_selected_country) #update selected country
        handleEnclaveNeighbours(selected_country)
        drag_start = values["graph"] #start dragging
    else: #if during drag movement
        drag_end = values["graph"]
        delta_x = drag_end[0] - drag_start[0]
        delta_y = drag_end[1] - drag_start[1]
        game_window["graph"].move(delta_x, delta_y) #move whole map
        offset_x = offset_x + delta_x #record offset in x direction
        offset_y = offset_y + delta_y #record offset in y direction
        drag_start = drag_end

#handle end of mouse click
def handleMouseUp():
    global drag_start
    if drag_start: #if dragging
        drag_start = False

#update selected country after key was pressed
def handleKeyPress(values):
    country = values["country_guess"]
    country_number = datamanager.getCountryNumber(country)
    if country_number:
        updateSelectedCountry(datamanager.getCountryName(country_number))
    else:
        deselectCountry()

#show final results
def showGameResults():

    if player_score > ai_score:
        result_text = "Congratulations! You won with a score of " + str(player_score) + " compared to the AIs " + str(ai_score)
    elif player_score < ai_score:
        result_text = "Looks like this round goes to the machine. The AI won with a score of  " + str(ai_score) + " compared to yours " + str(player_score)
    else:
        result_text ="A draw! You are truly on par with the AI, both of you reached a score of " + str(player_score)

    print(result_text)

    result_layout = [[sg.Text(result_text)]]

    result_window = sg.Window("Results",result_layout,resizable=True,finalize=True)

    result_window.bring_to_front()

    #result_window.read()

#return to the starting view
def backToMenu():

    global window,game_window,round_window

    window.close()
    round_window.close()

    window = sg.Window("Country Guesser", getDefaultlayout(), resizable=True)

    round_window = None




#show results of a game round
def showRoundResults(score):

    global round_window, rounds_left

    if round_window:
        round_window.close()

    searched_text = "The searched country was " + searched_country
    if score > 0:
        score_text = "You guessed the country at guess " + str (1 + len(guessed_countries)) + " and got " + str(score) + " points"
    else:
        score_text = "You didnt guess the right country and got no points"

    print (searched_text)
    print (score_text)

    if rounds_left > 1:
        button_text = "Play next Round"
    else:
        button_text = "Back to Menu"

    round_layout = [[sg.Text(searched_text)],
                    [sg.Text(score_text)],
                    [sg.Text("AI is making a guess:")],
                    [sg.Multiline(size=(60,20),disabled=True, echo_stdout_stderr=True, reroute_stdout=True,autoscroll=True)],
                    [sg.Text(key="ai_result")],
                    [sg.Button(button_text,visible=False,key="next_round")]]


    round_window = sg.Window("Round Results", round_layout, resizable=True, finalize=True)

    round_window.read(timeout=1000)

#continuously running loop
def eventLoop():

    global event,values

    while True:

        if window:
            event, values = window.read(timeout=30) #wait for events from default or image viewer window

            if event == "random_coordinates":
                selectRandomCoordinates(values["random_mode"])
            if event == "fetch_images":
                fetchAndUpdate(values["lang"])
            if event == "ai_guess":
                updateSettings(values)
                event, values = window.read(timeout=5) #get values
                getMask(values)
                aiGuess(mask)
            if event == "load_sky":
                loadImages(values["file_path"], 0)
            if event == "load_horizon":
                loadImages(values["file_path"], 1)
            if event == "play":
                enterPlayMode(values)
            if event == "show_map":
                game_window.Normal()
                game_window.bring_to_front()
            if "graph" in event:
                displayClickedImage(event,values)

        if game_window:

            event, values = game_window.read(timeout=100) #wait for events from game window

            if event == "MouseWheel:Up":
                if last_click:
                    zoomMap(last_click, 0.5, values)

            if event == "MouseWheel:Down":
                if last_click:
                    zoomMap(last_click, 2, values)

            if event == "graph": #mouseclick
                handleMapClick(values)

            if event == ("graph+UP"): #end of mouseclick
                handleMouseUp()

            if event == "show_images":
                window.bring_to_front()
                game_window.Minimize()

            if event == "guess":
                if not "not" in status:
                    if (selected_country == searched_country):
                        score = updatePlayerScore()
                        nextRound(score)
                    elif len(guessed_countries) + 1 < total_guesses:
                        nextGuess()
                    else:
                        nextRound(0)

            elif (len(event) == 1 or "BackSpace" in event): #key pressed
                handleKeyPress(values)

        if round_window:
            event, values = round_window.read(timeout=5)
            if event == "next_round":
                    round_window.Minimize()
                    if rounds_left == 0:
                        backToMenu()


#open window and wait for modules to load before entering the event loop
window = sg.Window("Country Guesser", getDefaultlayout(),resizable=True)
game_window = None
round_window = None
window.read(timeout=100)
print ("loading modules:")
import countryguesser
import datamanager
import imageeditor
print ("ready")
countries = datamanager.getAllCountriesWithExternalData()
eventLoop()


