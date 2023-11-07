#Library imports
import numpy as np
import streamlit as st
import cv2
import tensorflow
import keras
from keras.models import load_model


#Loading the Model
model = load_model('plant_disease.h5')

#Name of Classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

#Setting Title of App
st.image("plant.jpg", width = 600, )
html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:red;text-align:center;">Crop Disease Detection AI App </h2>
    </div>
    """
st.title("Crop Disease Detection")
st.markdown("This detection is just for common rust in corn, Early blight in Potato and Bacterial spot on tomatoes")
st.markdown("NOTE: This model is not trained for the crops without the disease. This model is just for my knowledge and research. Thank you")
st.markdown("Upload an image of the crop")

#Uploading the dog image
plant_image = st.file_uploader("Choose an image...", type="jpeg")
submit = st.button('Predict')
#On predict button click
if submit:


    if plant_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is "+result.split('-')[0]+ " crop with " + result.split('-')[1]))
        print(result)


if st.button("About Author"):
               st.text("Name : Osifalujo Temitope") 
               st.text("Email : temitopeosifalujo@gmail.com") 
               