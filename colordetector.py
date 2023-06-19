import datamanager
import imageeditor

#generate average histogram (or image) and compare it with the average histograms (or images) for all countries
def analyseHistograms(images,number=100,fov=90,mode=0,range=5,distance_mode=1):

     print ("analysing colors")

     countries = datamanager.getAllCountriesWithExternalData()

     similarities = []

     if mode == 0: #histograms
          histo = imageeditor.getAverageHistogram(images)
     elif mode == 1: #average images
          image = datamanager.packImageDict(imageeditor.mergeImagesRGBNumpy(images), (0, 0), 0, 0, 0)

     for country in countries: #get list with similarities between images and country
          if mode == 0:
               average_histo = datamanager.loadAverageHistogram(country,number,fov)
               if average_histo:
                    similarities.append([country,1 - imageeditor.compareHistogram(histo,average_histo,range)])
          elif mode == 1:
               average_image = datamanager.loadAverageImage(country,number,fov)
               if average_image:
                    if distance_mode == 0:
                         similarities.append([country, 1 - imageeditor.compareImagesAvergeDifferenceRGB(image,average_image)])
                    elif distance_mode == 1:
                         similarities.append([country, 1 - imageeditor.compareImagesNormalizedRootMeanSquareError(image,average_image)])
                    elif distance_mode == 2:
                         similarities.append([country, imageeditor.compareImagesStructuralSimilarity(image,average_image)])

     similarities.sort(key=lambda x: x[1], reverse=True) #sort the list

     #print ("colors similar to :", similarities)

     return similarities

