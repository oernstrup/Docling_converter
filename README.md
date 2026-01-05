# Docling_converter
- mkdir "folder"
-  cd "folder"
-  python -m venv .venv
-  .\\.venv\Scripts\activate
-  pip install -r requirements.txt
-  streamlit run app.py
-  Goto http://localhost:8501/
-  add a PDF document

-  <img width="569" height="414" alt="image" src="https://github.com/user-attachments/assets/0f6ac77a-6345-4924-93bc-067619b47309" />

---------------


Alternative usage ( offline models) (Not sure it works) 

- pip install hf_xet
- python -m pip install easyocr
- docling-tools models download --all -o ./models
- 
-  streamlit run .\app2.py



--------------------

Image support 
-  streamlit run '.\app3 - images ref.py'


-----------------------


Vision Support 
- mkdir /blip-image-captioning-base
- py .\download_blip.py
- pip install -U huggingface_hub
- pip install transformers -U



