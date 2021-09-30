import numpy as np
import cv2
import streamlit as st
from PIL import Image
from keras.models import model_from_json
from keras.preprocessing import image

### Excluding Imports ###
st.title("Brain Tumor Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    test_image = image.load_img(uploaded_file, target_size=(64,64))
    st.image(test_image, caption='Uploaded Image.', width=256)
    st.write("")
    st.write("Classifying...")
    
    test_image = test_image.resize((64,64), resample=Image.ANTIALIAS)
    
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    
    # load json and create model
    json_file = open('brain_tumor_dataset/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("brain_tumor_dataset/model.h5")
    
    
    label = loaded_model.predict_classes(test_image)
    print(label[0])
    if label[0] == 0:
        st.write(label[0],"- No Brain Tumor")
    else:
        st.write(label[0],"- Having Brain Tumor")

        
        