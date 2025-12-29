import PIL.Image
import streamlit as st
st.title("cifar 10 resnet_18 model Inference")
st.write("please upload your image")





col1, col2= st.columns([2,2])
col3, col4= st.columns([2,2])

with col3:
    uploaded_image = st.file_uploader("upload an image")

if uploaded_image:
    with col1:
        st.image(uploaded_image)

from PIL import Image
import numpy as np



def load_image(image_name):
    image = Image.open(image_name).resize(size= (32,32)).convert("RGB")

    image = np.array(image)/255

    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, 3)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 1, 3)
    
    # 4. Apply Normalization
    image = (image - mean) / std

    image = np.moveaxis(image,-1,0)[None,...].astype(np.float32)

    return image

@st.cache_resource
def load_model():
    import onnxruntime as ort
    session = ort.InferenceSession('final_model_aws.onnx',providers=['CPUExecutionProvider'])

    return session

def run_inference(session,image):
    import numpy as np
    output = session.run(output_names=None,input_feed={'input':image})
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    max_idx = int(np.argmax(output[0],axis=1))
    prediction=  classes[max_idx]
    
    return prediction

if st.button('Run Inference'):
    session = load_model()
    image = load_image(uploaded_image)
    prediction = run_inference(session,image)
    
    with col2:
        st.write(prediction)
