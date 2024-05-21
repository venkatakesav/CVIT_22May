import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import label
import os
import statistics
from sklearn.neighbors import KNeighborsClassifier
import time

def crop_handwritten_region(imgpath):
    
    img = cv2.imread(imgpath)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret3,bin_img = cv2.threshold(imgray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    width_threshold = 1000
    height_threshold = 500

    width_array =[]
    y_array =[]
    # Detect the three horizontal black lines
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        if w > width_threshold:
            if h <height_threshold:
                width_array.append (w) 
                y_array.append(y)
                
                
    indixes = sorted(range(len(width_array)), key=lambda k: width_array[k])
    indixes.reverse()
    three_lines_y=[]
    three_lines_y.append(y_array[0])
    three_lines_y.append(y_array[1])
    three_lines_y.append(y_array[2])
    three_lines_y.sort()
    newCooriate_y1= three_lines_y[1]
    newCooriate_y2= three_lines_y[2]

    cropped_imagebin = bin_img[newCooriate_y1+4:newCooriate_y2 , :]
    cropped_imagegray = imgray[newCooriate_y1+4:newCooriate_y2 , :]
    
    # array contains summation of black pixels on each row of the image
    sum_black_in_row = np.sum(cropped_imagebin < 255, axis=1)
    # threshold for rows contains black pixel > 15 
    intial_lines = sum_black_in_row > 15
    i = 0
    upp=0
    downn=0
    while i < len(intial_lines):
        if intial_lines[i] == True:
            begin_row = i
            if begin_row - 6 < 0:
                up = 0
            else:
                up = begin_row - 6
                if i==0:
                    upp=up
            while i < len(intial_lines) and intial_lines[i]:
                i += 1
            if i+5 > len(intial_lines) - 1 :
                down=len(intial_lines) - 1
            else:
                 down = i + 6
            if i - begin_row > 20:  # threshold for # of rows to be higher than 20 row 
                downn =down
        i += 1
        cropped_image_more = cropped_imagegray[upp:downn+4 , :]
    return cropped_imagebin, cropped_imagegray,cropped_image_more

def split_lines(img):
    
    # array contains summation of black pixels on each row of the image
    sum_black_in_row = np.sum(img < 255, axis=1)
    # threshold for rows contains black pixel > 15 
    intial_lines = sum_black_in_row > 15
    lines = []
    i = 0

    while i < len(intial_lines):
        if intial_lines[i] == True:
            begin_row = i
            if begin_row - 6 < 0:
                up = 0
            else:
                up = begin_row - 6
            while i < len(intial_lines) and intial_lines[i]:
                i += 1
            if i+5 > len(intial_lines) - 1 :
                down=len(intial_lines) - 1
            else:
                 down = i + 6
            if i - begin_row > 20:  # threshold for # of rows to be higher than 20 row 
                lines.append(img[up:down, :])
        i += 1
    return lines
# change this later to be R = 3 instead of R = 1
def lbp_calculate_pixels(img, x, y):
    threshold = img[x, y]
    bin_val = []
    bin_val.append(int(img[x - 3, y] >= threshold))
    bin_val.append(int(img[x - 2, y + 2] >= threshold))
    bin_val.append(int(img[x, y + 3] >= threshold))
    bin_val.append(int(img[x + 2, y + 2] >= threshold))
    bin_val.append(int(img[x + 3, y] >= threshold))
    bin_val.append(int(img[x + 2, y - 2] >= threshold))
    bin_val.append(int(img[x, y - 3] >= threshold))
    bin_val.append(int(img[x - 2, y - 2] >= threshold))
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    
    decimal = 0
    
    for i in range(len(power_val)):
        decimal += bin_val[i] * power_val[i]
    
    return decimal

def lbp_get_result(img):
    height, width = img.shape
    result = np.zeros((height, width), np.uint8)
    result = np.copy(img)
    for i in range(3, height-3):
        for j in range(3, width-3):
            result[i, j] = lbp_calculate_pixels(img, i, j)
    return result
# get the histogram of the resulted lbp img as our featured vector
def lbp_hist(img_lbp):
    img_reshaped = img_lbp.reshape(-1, 1)
    result_hist, result_hist_bins = np.histogram(img_reshaped)
    return result_hist

def lbp_normalize(lbp_hist):
    lbp_mean = statistics.mean(lbp_hist)
    lbp_hist = lbp_hist / lbp_mean
    return lbp_hist
path = 'E:\\Writer-Identification-System\\IAMdataset'
directory = os.listdir(path)

test_results = [] # holder for results to write after finishing
test_times = [] # holder for test times to write after finishing

for folder in directory:
    training_features = []
    labels = []
    test_features = []
    if '.txt' in (path + '\\' +folder + '\\'):
            continue
    for file in range(1,4):
        for img in range(1,3):
#             print(path + '\\' +folder + '\\' + str(file) + '\\' + str(img) +'.png')    
#             print(file)
#             labels.append(file)
            cropped_imgbin,cropped_imggray, cropped_img_more = crop_handwritten_region(path + '\\' +folder + '\\' + str(file) + '\\' + str(img) +'.png')
            for i in split_lines(cropped_imgbin):
                lbp_img = lbp_get_result(i)
                result_hist = lbp_hist(lbp_img)
                result_normalized = lbp_normalize(result_hist)
                training_features.append(result_normalized)
                labels.append(file)
    classifier = KNeighborsClassifier(n_neighbors=5)  
    classifier.fit(training_features, labels) 
    
#     print(path + '\\' +folder + '\\test.png')
    cropped_imgbin,cropped_imggray, cropped_img_more = crop_handwritten_region(path + '\\' +folder + '\\test.png')
    start_time = time.time()   
    for i in split_lines(cropped_imgbin):
#         pred = []
        lbp_img = lbp_get_result(i)
        result_hist = lbp_hist(lbp_img)
        result_normalized = lbp_normalize(result_hist)
        test_features.append(result_normalized)

#         p = classifier.predict_proba(result_normalized.reshape(1, -1))
#         p = np.sum(p, axis=0)
#         r = classifier.classes_[np.argmax(p)]
#         print(r)
    
    
    writer_prediction = classifier.predict(test_features)
    end_time = time.time()
    run_time = end_time - start_time
    test_times.append(run_time)
    x = np.array(writer_prediction) 
#     print("Most frequent value in the above array:") 
    print(np.bincount(x).argmax())
    test_results.append(np.bincount(x).argmax())
#     print("knn",writer_prediction)

## output writing in files
results_writer = open(os.path.join(path,'results.txt'),'w')
for result in test_results:
    results_writer.write(str(result)+'\n')

results_writer.close()

times_writer = open(os.path.join(path,'time.txt'),'w')
for i in range(len(test_times)):
    times_writer.write(str(test_times[i])+ '\n')
times_writer.close()