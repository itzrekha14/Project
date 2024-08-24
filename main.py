# ======================= IMPORT PACKAGES 
import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)

import base64

# --------------- BACKGROUND IMAGE -----------


st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Drug Discovey identification using ML and DL"}</h1>', unsafe_allow_html=True)


''''''''''''''''''''''''''''''
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.jpg') 

print('\n')
print('\t\t','********************   READ DATA   ********************')
print('\n')


# Read input images and assign labels based on folder names
print(os.listdir("Dataset/"))

SIZE = 65  #Resize images

#Capture training data and labels into respective lists
train_images = []
train_labels = [] 

for directory_path in glob.glob("Dataset/train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
#        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

#Convert lists to arrays        
train_images = np.array(train_images)
train_labels = np.array(train_labels)


# Capture test/validation data and labels into respective lists

test_images = []
test_labels = [] 
for directory_path in glob.glob("Dataset/test/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)

#Convert lists to arrays                
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#********************************************************************************************************************#
print('\n')
print('\t\t','********************  LABEL ENCODING ********************')
print('\n')
#Encode labels from text to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)
print("Test Label encoding",test_labels_encoded,"\n")
print("Train Label encoding",train_labels_encoded)

#***************************************************************************************************#
##############SPLITTING DATASET####################

#Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

#One hot encode y values for neural network. 
from keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#***************************************************************************************************#

print('\n')
print('\t\t','************* SUPPORT VECTOR MACHINE (SVM) ***************','\n')


st.write('\n')
st.write('\t\t','************* SUPPORT VECTOR MACHINE (SVM) ***************','\n')

x_train1=x_train[:,:,1,1]
x_test1=x_test[:,:,1,1]

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

clf = SVC()
clf.fit(x_train1, y_train) 
svm_pred=clf.predict(x_test1)

svm_accu=accuracy_score(y_test,svm_pred)*100
print("\n"," SVM Accuracy : ",svm_accu,'%',"\n")


st.write("\n"," SVM Accuracy : ",svm_accu,'%',"\n")

svm_cr=classification_report(y_test,svm_pred)
print("Classification Report: \n",svm_cr)

st.write("Classification Report: \n",svm_cr)

svm_cm = confusion_matrix(y_test, svm_pred)
  
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.matshow(svm_cm, cmap=plt.cm.BrBG, alpha=0.3)
for i in range(svm_cm.shape[0]):
    for j in range(svm_cm.shape[1]):
        ax.text(x=j, y=i,s=svm_cm[i, j], va='center', ha='center', size='xx-large')
print('Confusion Matrix :')
plt.show()

#***************************************************************************************************#

print('\n')
print('\t\t','*******  CONVOLUTION NEURAL NETWORK (CNN)  *******')
print('\n')



st.write('\n')
st.write('\t\t','*******  CONVOLUTION NEURAL NETWORK (CNN)  *******')
st.write('\n')



from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(32, 2, activation="relu", input_shape=(65,65,3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, 2, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(4, activation = 'sigmoid'))
model.compile(loss = 'categorical_crossentropy',optimizer = "adam",metrics = ['accuracy'])
model.summary()


model.fit(x_train, y_train_one_hot, batch_size=1,epochs=3, verbose=1)


Y_pred=model.predict(x_test)
# Y_p=model.predict_classes(x_test)
#Y_pd=np.round(abs(Y_p))
# cm = confusion_matrix(y_test, Y_pred)

acc = model.evaluate(x_train, y_train_one_hot)[1]*100
print('\n'," ACCURACY : ", acc,'%')

st.write('\n'," ACCURACY : ", acc,'%')

#***************************************************************************************************#
print('\n')
print('\t\t','********** ACCURACY COMPARISON  **********')
print('\n')

import seaborn as sns
sns.barplot(x=["CNN","SVM"],y=[acc,svm_accu])
plt.title("Comparison")
# plt.savefig("Gra.png")
plt.show()


st.image("Gra.png")

#***************************************************************************************************#



# ========== PREDICTION

st.markdown(f'<h1 style="color:#000000;font-size:24px;">{" Prediction "}</h1>', unsafe_allow_html=True)

  
import os 

from sklearn.model_selection import train_test_split

d1 = os.listdir('Data/amaxicilin')

d2 = os.listdir('Data/clonazepam')

d3 = os.listdir('Data/Gelus')

d4 = os.listdir('Data/paracetamol')


import matplotlib.image as mpimg  
  
dot1= []
labels1 = []
for img1 in d1:
        # print(img)
        img_1 = mpimg.imread('Data/amaxicilin/' + "/" + img1)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(0)

for img1 in d2:
        # print(img)
        img_1 = mpimg.imread('Data/clonazepam/' + "/" + img1)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(1)        


for img1 in d3:
        # print(img)
        img_1 = mpimg.imread('Data/Gelus/' + "/" + img1)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(2)        


for img1 in d4:
        # print(img)
        img_1 = mpimg.imread('Data/paracetamol/' + "/" + img1)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(3) 


uploaded_file = st.file_uploader("Upload Image")


# aa = st.button("UPLOAD IMAGE")

if uploaded_file is None:
    
    st.text("Please upload an image")

else:
    import numpy as np

    img = mpimg.imread(uploaded_file)
    st.text(uploaded_file)
    st.image(img,caption="Original Image")
    
    

    #============================ PREPROCESS =================================
    
    #==== RESIZE IMAGE ====
    
    resized_image = cv2.resize(img,(300 ,300))
    img_resize_orig = cv2.resize(img,((50, 50)))
    
    fig = plt.figure()
    plt.title('RESIZED IMAGE')
    plt.imshow(resized_image)
    plt.axis ('off')
    plt.show()
    st.image(resized_image,caption="Resized Image")

       
             
    #==== GRAYSCALE IMAGE ====
    
    
    
    SPV = np.shape(img)
    
    try:            
        gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
        
    except:
        gray1 = img_resize_orig
       
    fig = plt.figure()
    plt.title('GRAY SCALE IMAGE')
    plt.imshow(gray1,cmap='gray')
    plt.axis ('off')
    plt.show()
    
    
aa = st.button("PREDICT")  
    
    
    
if aa:

    Total_length = len(d1) + len(d2) + len(d3) + len(d4)
    
    temp_data1  = []
    for ijk in range(0,Total_length):
        temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
        temp_data1.append(temp_data)
    
    temp_data1 =np.array(temp_data1)
    
    zz = np.where(temp_data1==1)
    
    if labels1[zz[0][0]] == 0:

        
        st.text('--------------------------------------')
        st.text("Identified = amaxicilin")
        st.text('-------------------------------------')
            
    elif labels1[zz[0][0]] == 1:

        
        st.text('--------------------------------------')
        st.text("Identified = clonazepam")
        st.text('-------------------------------------')

            
    elif labels1[zz[0][0]] == 2:

        
        st.text('--------------------------------------')
        st.text("Identified = Gelus")
        st.text('-------------------------------------')

    elif labels1[zz[0][0]] == 3:

        
        st.text('--------------------------------------')
        st.text("Identified = paracetamol")
        st.text('-------------------------------------')

#********************************************************************************************************************#





















