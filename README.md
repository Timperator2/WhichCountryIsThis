# WhichCountryIsThis

Official Git Repository for "Which Country is This"

Watch the AI guess location based on Google Street View imagery or challenge the system in game mode. 

You will need to provide a Google Street View API Key (just paste it into the datamanager.py file)

Run guimanager.py to open the program or watch the demo video to see it in action.

Should work with python 3.10 and the listed requirements. However, if you want to use a different cuda version for example, you have to make sure you install the right packages. Talking about cuda, while it is possible to run the program without graphics card, using one improves performance significantly.

Requirements:
opencv-contrib-python==4.5.4.60
opencv-python==4.5.4.60
opencv-python-headless==4.5.4.60
git+https://github.com/openai/CLIP.git
easyocr==1.4.2
geotext==0.4.0
huggingface_hub==0.7.0
langdetect==1.0.9
lingua-language-detector==1.0.1
numpy==1.23.0
Pillow==9.3.0
pycountry==22.3.5
PySimpleGUI==4.60.4
requests==2.27.1
scikit_image==0.19.2
scikit_learn==1.1.1
Shapely==1.8.2 
torch==1.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
tqdm==4.64.0
transformers==4.19.4
yolov5==6.1.6
requests==2.27.1

![Poster](CountryGuesser_Poster__ECIR_2023.png)

[Read the full paper](Which_Country_is_This___ECIR_23_Demo_.pdf)

[Published Version](https://link.springer.com/chapter/10.1007/978-3-031-28241-6_26)

[Watch the video](which_country_is_this_demo.mp4)


