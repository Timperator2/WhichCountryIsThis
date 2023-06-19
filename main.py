import datamanager as dm
import sundetector as su
import textandlanguagedetector as tl
import PySimpleGUI as sg



#getMostConfident("TestImages//Writing//test3sk.png")

# tl.changeReader(0)
# text = tl.filterResults(tl.readText("TestImages//la35.1494047lo129.0591608//heading180fov45pitch45.png"))
# print (text)
# print (tl.getConfidence(text))

#getMostConfident("TestImages//la35.1494047lo129.0591608//test3sk.png")
# result = []
# average_confidence = 0.0
# counter = 0
# for r in resultraw:
#     if r[2] > 0.03 and r[1]!="Google" and r[1]!="Gccgle" and r[1]!="Gocgle":
#         print ("character", r[1])
#         print ("confidence",r[2])
#         average_confidence = average_confidence + r[2]
#         counter = counter +1
#         try:
#             print(detect_langs(r[1]))
#         except:
#             print("lol")
#         result.append(r[1])
#
# print ("total confidence",average_confidence/counter)
#
# rtxt = ""
#
# for r in result: #adding or iterating
#     try:
#         rtxt = rtxt + r + " "
#     except:
#         print ("lol")
#
# print (rtxt)
# print (detect_langs(rtxt))



# images = getFullPanorama((35.1494047,129.0591608),x=800,y=800,fov=30)
#
# saveTestImages(images)

# images = loadTestImages((-34.5136739,-63.7089253 ),pitchs=(45,90),fovs=(45))
# print (getHemissphere(findSun(images)))
#
# images = loadTestImages(( -13.27692405368929,-43.66191066793041),pitchs=(45,90),fovs=(45))
# print (getHemissphere(findSun(images)))
#
# images = loadTestImages((-1.75084132153959,-48.18604139075298),pitchs=(45,90),fovs=(45))
# print (getHemissphere(findSun(images)))
#
# images = loadTestImages((-41.806465162,145.233951081238),pitchs=(45,90),fovs=(30))
# print (analyseSun(images,zoom_fov=10))


# images = loadTestImages((-41.806465162,145.233951081238),pitchs=(45,90),fovs=(30))
# print (analyseSun(images,zoom_fov=10,mode=1))

#
# images = loadTestImages((-16.66358903965267,-72.61103442963667 ),pitchs=(45,90),fovs=(45))
# print (getHemissphere(findSun(images)))
#
# images = loadTestImages((59.30945599306971,28.61245044397),pitchs=(45,90),fovs=(45))
# print (getHemissphere(findSun(images)))
#
# images = loadTestImages((-49.8389019858466,-69.04384172882675),pitchs=(45,90),fovs=(45))
# print (getHemissphere(findSun(images)))
#
# images = loadTestImages((-31.00234252425759,-50.83402090643757),pitchs=(45,90),fovs=(45))
# print (getHemissphere(findSun(images)))
#
# images = loadTestImages((-0.8766878671065835,101.3141824302687 ),pitchs=(45,90),fovs=(45))
# print (getHemissphere(findSun(images)))

# images = loadTestImages((-24.4163075,151.5503756),pitchs=(45,90),fovs=(45))
# print (getHemissphere(findSun(images)))

# images = scanSky(findAvailableRandomImage())
# saveTestImages(images)
# print (getHemissphere(findSun(images)))

# images = scanSky(findAvailableRandomImage())
# print (analyseSun(images))
# print (analyseSun(images,zoom_fov=10,mode=1))

# images = loadTestImages((51.9852938,-1.8254026))
# print (analyseSun(images))
# print (analyseSun(images,zoom_fov=15,mode=1))

# image = packImageDict(loadImage("testheading180fov90pitch0.png"),(0,0),180,90,0)
# images = []
# images.append(image)
# print (zoomedImagesFromExisting(images,30))










#-34.5136739,-63.7089253 good value for sun north
#-41.80646516235531,145.233951081238 good value for sun not to so quit north but should still work
#56.60159495087051,43.33342193213269 location on northern hemisphere where sun is not north or south
#46.05148781876623,-68.12085038418978 location on north globe where sun is on top
#-43.3953763,-70.8766089 edge case sun is in top (with 225 tresh) but could be north (very close value, maybe add another threshhold?: When Sun above is it north east? )
#60.36745238542875,15.70642903392583 sun is clearly south
#59.30945599306971,28.61245044397 sun is assumed to be northern on northern hemisphere
#-16.66358903965267,-72.61103442963667 wrong sun on southerhn hemi with fov 60
#-1.75084132153959,-48.18604139075298 same vegetation als problem?
#add up and see borders as solution? #0+60 60+120 120+180 180+240 240+300 300+0
                                      #0=n  #1=?    #2=s   #3=s   #4=?     #5=n

                        #for 45:  #0+45 45+90 90+135 135+180 180+225 225+270 270+315 315+0
                        #           n     n     ?      s        s      s       ?     n

#-18.35943147301478,-54.66502203262612 doesnt work
#-49.8389019858466,-69.04384172882675 too many clouds #filter with basic brightness?
#-13.27692405368929,-43.66191066793041 error
#-0.8766878671065835,101.3141824302687 error
#-18.61839279584864,-53.74440461773968 sun should be top
#-31.00234252425759,-50.83402090643757 approach without sum maybe too risky but other maybe too conservative
#problem: wolken heller als sonne?
#testen 60 fov 60 pitch
#idee: verschiedene ansätze gegegeneinander und trefferquote vergleichen dann entscheiden
#40.9192282,-95.9304548 sun should be visible south (might be top) okay it rly seems to be 90 lol
#-24.4163075,151.5503756 unclear south