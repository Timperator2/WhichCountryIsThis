import io
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import datamanager
import numpy
import  skimage.metrics
from sklearn.cluster import KMeans
from collections import Counter
import itertools

#split a larger image into several images with smaller fovs
def zoomedImagesFromExisting (images,zoom_fov):
    new_images = []
    for img in images:

        image_file = Image.open(io.BytesIO(img["file"]))
        steps = int (img["fov"]) / zoom_fov #calculate number of new images
        width, height = image_file.size

        width_cut = 0
        counter_heading = 0
        counter_pitch = 0
        while width_cut < width:
            new_width_cut = width_cut + width /steps
            height_cut = 0
            while height_cut < height: #get area
                new_height_cut = height_cut + height / steps
                area = (width_cut, height_cut, new_width_cut, new_height_cut)
                height_cut = new_height_cut
                new_heading = int (img["heading"]) - int (img["fov"]) / 2 + counter_heading * zoom_fov + zoom_fov / 2
                new_pitch = int (img["pitch"]) - int (img["fov"]) / 2 + counter_pitch * zoom_fov + zoom_fov / 2

                image_bytes = datamanager.ImageToBytes(image_file.crop(area)) #crop out area
                new_images.append(datamanager.packImageDict(image_bytes, img["coordinates"], new_heading, zoom_fov, new_pitch))

                counter_pitch = counter_pitch + 1
                #image_file.crop(area).show()

            width_cut = new_width_cut
            counter_heading = counter_heading + 1


    return new_images

#form a new image out of two images
def mergeTwoImages(image_left,image_right,mode):
    image_file_left = Image.open(io.BytesIO(image_left["file"]))
    image_file_right = Image.open(io.BytesIO(image_right["file"]))
    if mode == 0: #add horizontal
        new_width = image_file_left.width + image_file_right.width
        new_height = max(image_file_left.height, image_file_right.height)
        complete_image = Image.new("RGB",(new_width, new_height))
        complete_image.paste(image_file_left,(0,0))
        complete_image.paste(image_file_right,(image_file_left.width, 0))
    else: #add vertical
        new_width = max(image_file_left.width, image_file_right.width)
        new_height = image_file_left.height + image_file_right.height
        complete_image = Image.new("RGB",(new_width, new_height))
        complete_image.paste(image_file_left,(0,0))
        complete_image.paste(image_file_right, (0, image_file_left.height))

    #complete_image.show()
    return complete_image

#form a new image out of a larger number of images
def mergeAllImages(images,mode):

    new_width = 0
    new_height = 0

    for img in images: #create new image large enough
        try:
            image_file = Image.open(io.BytesIO(img["file"]))
        except:
            image_file = img

        if mode == 0: #horizontal
            new_width = new_width + image_file.width
            new_height = max(new_height,image_file.height)
        else: #vertical
            new_width = max(new_width, image_file.width)
            new_height = new_height + image_file.height

    complete_image = Image.new("RGB", (new_width, new_height))

    last_width = 0
    last_height = 0

    for index,img in enumerate (images): #fill the image
        try:
            image_file = Image.open(io.BytesIO(img["file"]))
        except:
            image_file = img

        if mode == 0: #horizontal
            complete_image.paste(image_file,(0+index*last_width,0))
            last_width = image_file.width
        else: #vertical
            complete_image.paste(image_file, (0, 0 + index * last_height))
            last_height = image_file.width

    #complete_image.show()

    return complete_image

#create a panorama out of several images
def mergePanorama(images):
    heading_levels = []
    heading_level = []
    current_heading = -1
    images.sort(key=lambda x: int (x["heading"])) #sort by heading
    for index,img in enumerate (images): #group images with same headings
        if img["heading"] != current_heading: #if image has a different heading
            if (current_heading != -1): #if not first iteration
                heading_levels.append(heading_level.copy()) #append heading level with all images
            if index < len(images) - 1: #if not last image
                current_heading =img["heading"] #heading of the current level
                heading_level = []
            else: #if last image
                heading_levels.append([img])

        heading_level.append(img) #append image to heading level


    merged_images = []

    for heading in heading_levels: #for all images with same heading
        heading.sort(reverse=True,key=lambda x: int(x["pitch"])) #sort by pitch
        merged_image = mergeAllImages(heading,1) #verticaly stich together
        merged_images.append(merged_image)

    panorama = mergeAllImages(merged_images,0) #horizontly stich together all verticaly stitched images

    #panorama.show()

    return datamanager.packImageDict(datamanager.ImageToBytes(panorama),images[0]["coordinates"],0,360,0)

#blur the two watermarks at the bottom of a street view image
def blurWatermark(image):

    img = Image.open(io.BytesIO(image["file"]))

    watermark_left_position = (0,img.height - 25,70,img.height)
    watermark_left = img.crop(watermark_left_position)
    watermark_left = watermark_left.filter(ImageFilter.GaussianBlur(radius = 5))
    img.paste(watermark_left,watermark_left_position)

    watermark_right_position = (img.width-70, img.height - 25, img.width, img.height)
    watermark_right = img.crop(watermark_right_position)
    watermark_right = watermark_right.filter(ImageFilter.GaussianBlur(radius=5))
    img.paste(watermark_right, watermark_right_position)

    #img.show()

    return datamanager.packImageDict(datamanager.ImageToBytes(img),image["coordinates"],image["heading"],image["fov"],image["heading"])

#resize a number of images
def resize_images(images,new_x,new_y):
    for image in images:
        img = Image.open(io.BytesIO(image["file"]))
        img = img.resize((new_x,new_y))
        image["file"] = datamanager.ImageToBytes(img)

    return images

#generate the "average image" of a number of images
def mergeImagesRGB(images,sharpen=False):
    rgb_data = [] #list to contain the summed up rgb data of all images
    width = 0
    height = 0
    for index,image in enumerate (images):
        img = Image.open(io.BytesIO(image["file"]))
        if sharpen:
            img = ImageEnhance.Contrast(img).enhance(2.0) #increase image contrast
        width,height = img.size
        if index == 0: #for the first image
            rgb_data = [[0 for x in range(width)] for y in range(height)] #fill the list with zeros
        for w in range(width):
            for h in range(height):
                if (index == 0): #for the first image
                    rgb_data[w][h] = img.getpixel((w,h)) #fill the list with the rgb values for each pixel
                else:
                    color = [0,0,0]
                    color[0] = img.getpixel((w,h))[0] + rgb_data[w][h][0]
                    color[1] = img.getpixel((w,h))[1] + rgb_data[w][h][1]
                    color[2] = img.getpixel((w,h))[2] + rgb_data[w][h][2]
                    rgb_data[w][h] = color #set list values to the sum of old and new rgb values

    average_image = Image.new("RGB",(width,height)) #create a new image
    for w in range(width):
        for h in range(height):
            color = [0, 0, 0]
            color[0] =  int (rgb_data[w][h][0] / len(images))
            color[1] = int (rgb_data[w][h][1] / len(images))
            color[2] = int (rgb_data[w][h][2] / len(images))

            average_image.putpixel((w,h),tuple(color)) #color the image with the average pixel rgb values

    #average_image.show()

    return datamanager.ImageToBytes(average_image)

#generate the "average image" of a number of images with numpy (much faster)
def mergeImagesRGBNumpy(images,sharpen=False):

    if not sharpen: #increase image contrast (when selected) and create a numpy array from images
        images = numpy.array([numpy.array(Image.open(io.BytesIO(image["file"]))) for image in images])
    else:
        images = numpy.array([numpy.array(ImageEnhance.Contrast(Image.open(io.BytesIO(image["file"]))).enhance(2.0)) for image in images])

    average_values = numpy.average(images, axis=0) #get the average rgb values for the numpy array

    average_image = Image.fromarray(average_values.astype('uint8')) #create a new image from the numpy array

    #average_image.show()
    #datamanager.saveTempImage(average_image)

    return datamanager.ImageToBytes(average_image)

#form image color clusters via k-means
def getImageCluster(images,sharpen=True):

    if sharpen:
        images = numpy.array([numpy.array(ImageEnhance.Contrast(Image.open(io.BytesIO(image["file"]))).enhance(2.0)) for image in images])
    else:
        images = numpy.array([numpy.array(Image.open(io.BytesIO(image["file"]))) for image in images])

    images = images.reshape(-1,3) #reshape the arry for k-means

    return kmeansColors(images) #cluster

#apply k-means to get different clusters based on colors
def kmeansColors(images):

    overview = []
    overview_element = [[0,0,0],0]

    k_means = KMeans(n_clusters=4)  # number of clusters too form (higher numbers equals to many shades of grey)
    k_means.fit(images)

    labels = k_means.labels_#the assigned clusters for each pixel
    colors = k_means.cluster_centers_ #colors are centers
    all_cluster = Counter(labels)

    for cluster in all_cluster:
        overview_element[0] = colors[cluster] #get the dominant color
        overview_element[1] = all_cluster[cluster] / len(labels) #get percentage of pixels belonging to color
        overview.append(overview_element.copy())

    return overview

#from cluster based on cluster of single images
def clusterImageCluster(images,sharpen=True,weighted=True):

    merged_data = []

    for image in images:
        cluster = getImageCluster([image],sharpen) #get cluster for single images
        for color in cluster:
            if not weighted:
                color[1] = 0.1 #every color is later append equally often
            for i in range(int(color[1] * 100)):
                merged_data.append(color[0]) #append all dominant colors

    return kmeansColors(merged_data) #final clustering

#get the average histogram of a set of images
def getAverageHistogram(images):

    total_histo = []

    for image in images:
        img = Image.open(io.BytesIO(image["file"]))
        histo = img.histogram() #get the histogram

        if (len(total_histo) == 0): #if first histogram
            total_histo = histo
        else:
            for index,value in enumerate(histo):
                total_histo[index] = total_histo[index] + value #add the values for each interval

    for index,value in enumerate(total_histo):
        total_histo[index] = value / len(images) #get the average value for each interval

    return total_histo

#compare two histograms
def compareHistogram(histogram,test_histogram,steps = 0):
    total_difference = 0
    channel_start = 0
    channel_end = 255
    for index,value in enumerate (histogram):
        if steps == 0: #only compare equal intervalls between the two histograms
            total_difference = total_difference + abs (value - test_histogram[index])

        elif index >= channel_start and index < channel_end - steps: #when steps still in range of channel
            stepped_histo = 0
            stepped_test_histo = 0
            for step in range((steps * 2) + 1): #sum up and compare a range of intervals
                stepped_histo = stepped_histo + histogram[index -steps + step]
                stepped_test_histo = stepped_test_histo + test_histogram[index -steps + step]

            total_difference = total_difference + abs(stepped_histo/(steps*2) - stepped_test_histo/(steps*2) ) #get the average distance

        elif channel_start < 510: #when steps out of channel range
            channel_start = channel_end #move to next channel
            channel_end = channel_end + 255

    maximum_difference = 0
    for val in histogram:
        maximum_difference = maximum_difference + val #get total number of color values in one histogram

    maximum_difference = maximum_difference * 2 #the maxiumum possible difference is two times the total number of color values

    return (total_difference / (maximum_difference )) #normalize

#get the average difference in rgb values for each pixel
def compareImagesAvergeDifferenceRGB(image,test_image):

    difference = [0,0,0]

    img = Image.open(io.BytesIO(image["file"]))
    test_img = Image.open(io.BytesIO(test_image["file"]))

    width, height = img.size

    for w in range(width): #iterate over all pixels and get the difference for each color channel
        for h in range(height):
            difference[0] = difference[0] + abs(img.getpixel((w,h))[0] - test_img.getpixel((w,h))[0])
            difference[1] = difference[1] + abs(img.getpixel((w, h))[1] - test_img.getpixel((w, h))[1])
            difference[2] = difference[2] + abs(img.getpixel((w, h))[2] - test_img.getpixel((w, h))[2])


    difference[0]= difference[0] / (width * height)
    difference[1]=difference[1] / (width * height)
    difference[2]=difference[2] / (width * height)

    return ((difference[0] + difference[1] + difference[2]) / 3) / 255 #return the average difference between 0 and 1

#get the normalized root mean square error between two images
def compareImagesNormalizedRootMeanSquareError(image, test_image):
    img = numpy.asarray(Image.open(io.BytesIO(image["file"])))
    test_img = numpy.asarray(Image.open(io.BytesIO(test_image["file"])))

    return skimage.metrics.normalized_root_mse(img,test_img)

#get the structural similarity between two images
def compareImagesStructuralSimilarity(image, test_image):

    img = numpy.asarray(Image.open(io.BytesIO(image["file"])))
    test_img = numpy.asarray(Image.open(io.BytesIO(test_image["file"])))

    return skimage.metrics.structural_similarity(img,test_img,channel_axis=2)

#compare the cluster of two images
def compareImageCluster(all_cluster, all_test_cluster,weighted=True):

    min_distances = []

    for index, cluster in enumerate(all_cluster): #for all cluster from one image
        center = numpy.array (cluster[0]) #get the dominant color
        distances = []
        for second_index,test_cluster in enumerate (all_test_cluster): #for all cluster from the other image
            test_center = numpy.array (test_cluster[0]) #get the dominant color
            color_difference = numpy.linalg.norm(center-test_center) / 442 #nornmalized euclidian distance between dominant colors
            if weighted: #consider number of pixels per cluster
                size_difference = abs (test_cluster[1] - cluster[1])
                color_difference = (color_difference + size_difference) / 2

            distances.append(color_difference) #append distance to a list

        min_distances.append(min(distances)) #find and append the shortest detected distance

        total_min_distance = 0
        for min_distance in min_distances: #sum up all minimum distances
            total_min_distance = total_min_distance + min_distance


        return (total_min_distance / len(all_cluster)) #get the average length of the shortest distances

#get the image which represents a set of images most
def mostSimilarImage(images):
    difference = [[0,0],0]
    differences = []

    for a, b in itertools.combinations(images, 2): #check every possible image combination
        difference[0]=[images.index(a),images.index(b)] #get the index of both images
        difference[1]=compareImagesNormalizedRootMeanSquareError(a,b) #get their difference
        differences.append(difference.copy())

    average_differences = []

    for index in range(len(images)): #for each image
        total_difference = 0
        for difference in differences:
            if index in difference[0]:
                total_difference = total_difference + difference[1] #sum up the distances to all other images

        average_differences.append(total_difference / len(images))

    most_similar = images[average_differences.index(min(average_differences))] #get the image with the shortest average distance

    return most_similar

#crop an object given its bounding box
def cropObject(image,boxes):

    img = imageToPIL(image)
    img = img.crop(boxes)

    #img.show()
    datamanager.saveTempImage(img)

    return img

#highligh the blurred parts of an image
def highlightBlur(image):

    img = image

    img = img.convert("L") #make image grayscale

    #datamanager.saveTempImage(img)
    #img.show()

    blurred = numpy.asarray(img.filter(ImageFilter.GaussianBlur(radius=5))) #blur the image

    #datamanager.saveTempImage(img.filter(ImageFilter.GaussianBlur(radius=5)))
    #img.filter(ImageFilter.GaussianBlur(radius=5)).show()

    original = numpy.asarray(img) #convert to array for mathematical operations

    result = original - blurred #substract blurred image from original
    result = Image.fromarray(result)

    #datamanager.saveTempImage(result)
    #result.show()

    result = result.filter(ImageFilter.GaussianBlur(radius=3)) #blur again

    #datamanager.saveTempImage(result)
    #result.show()

    width, height = result.size
    for w in range(width): #filter blurred pixels
        for h in range(height):
            if result.getpixel((w, h)) < 50:
                result.putpixel((w, h), 255)
            else:
                result.putpixel((w, h), 0)

    #datamanager.saveTempImage(result)
    #result.show()

    return result

#draw cantours in an image
def drawContours(image,coords):

    img = image
    draw = ImageDraw.Draw(img, img.mode)
    draw.polygon(coords, fill=None) #draw contours as polygon

    #datamanager.saveTempImage(img)
    #img.show()

    return img

#split the image into several tiles
def tileImageNumpy(image,M,N):

    image = numpy.asarray(image)

    return [image[x:x + M, y:y + N] for x in range(0, image.shape[0], M) for y in range(0, image.shape[1], N)]



#convert an image to the PIL format
def imageToPIL(image):
    try:
        return Image.open(io.BytesIO(image["file"]))
    except:
        return image


#check if a color is present in an image
def checkForColor(image,color,threshold):

    image = imageToPIL(image).convert("HSV") #convert to HSV space

    #image.show()


    number = 0

    width,height = image.size
    for w in range(width):
        for h in range(height):
            val = image.getpixel((w, h))
            if val[1] <= 50: #if saturation is too low
                image.putpixel((w, h), (0, 0, 0)) #filter
            elif (val[0] <= color[0] or val[0] >= color[1]): #if pixel is not in color range
                image.putpixel((w,h),(0,0,0)) #filter
            else:
                number = number +1

    image.show()
    #datamanager.saveTempImage(image.convert("RGB"))

    percentage = number / (image.width * image.height) #get percentage of pixels with the color

    if percentage > threshold:
        return percentage
    else:
        return 0

#check if a color is present in an image and where
def checkForColorAndSides(image,color,threshold):

    image = numpy.asarray(imageToPIL(image).convert("HSV")) #convert to hsv
    left = 0
    middle = 0
    right = 0

    for height in image:
        for position,width in enumerate (height):
            if width[0] >= color[0] and width[0] <= color[1] and width[1] >= 50: #count pixels for region
                if position < len(height) / 3:
                    left = left +1
                elif  position <= (len(height) / 3) * 2:
                    middle = middle +1
                else:
                    right = right +1


    total = left + middle + right

    percentage = total / (image.shape[0] * image.shape[1])

    if percentage > threshold: #if total number of pixel with colors is above threshold
        return [percentage,left/total,middle/total,right/total] #return pixels by regions
    else:
        return [0,0,0,0]









