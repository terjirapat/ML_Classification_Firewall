# 🛠️ THNumber_img_classification ( without Deep Learning )
:pushpin: **Goal** :fire: : <br>
>การสอน computer ให้สามารถแยกแยะรูปภาพของเลขไทย ซึ่งเขียนด้วยลายมือ :crayon: ( ขนาด 28x28 pixels ) ว่าเป็นเลขอะไร <br>
>ด้วยการใช้เครื่องมือ ```Machine Learning``` ( โดยที่ **ไม่** มีการใช้ ```Neural Network``` หรือ ```Deep learning``` ในการ Train ) <br> 
>
>และสร้าง Application :toolbox::wrench: สำหรับคนที่ไม่สามารถเขียน Code ในการทำ ```Machine Learning (ML)``` ดังกล่าว ให้สามารถ Train ```Machine Learning``` ผ่าน App ได้ <br>

# <h3> Topics </h3>
สำหรับ Project นี้ เราจะแบ่งออกเป็น 2 หัวข้อ ได้แก่ <br>

>- [การทำ Machine Learning (ML)](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-#%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%97%E0%B8%B3-machine-learning-ml)
>- [การทำ Application :toolbox::wrench:](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-#%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%97%E0%B8%B3-application-toolboxwrench)

โดยจะเริ่มหัวข้อตามลำดับ <br>

# <h4>Languages & Tools</h4>

[![](https://img.shields.io/badge/code-python3.9-green?style=f?style=flat-square&logo=python&logoColor=white&color=2bbc8a)](https://www.python.org/)
[![](https://img.shields.io/badge/tools-jupyter-orange?style=f?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![](https://img.shields.io/badge/tools-dash-green?style=f?style=flat-square&logo=plotly&logoColor=white&color=2bbc8a)](https://dash.plotly.com/)
[![](https://img.shields.io/badge/tools-SkLearn-green?style=f?style=flat-square&logo=scikitlearn&logoColor=white&color=2bbc8a)](https://scikit-learn.org/stable/)
[![](https://img.shields.io/badge/tools-VSCode-blue?style=f?style=flat-square&logo=visualstudiocode&logoColor=white)](https://code.visualstudio.com/)
[![](https://img.shields.io/badge/tools-Pandas-green?style=f?style=flat-square&logo=pandas&logoColor=white&color=2bbc8a)](https://pandas.pydata.org/)
[![](https://img.shields.io/badge/tools-Numpy-green?style=f?style=flat-square&logo=numpy&logoColor=white&color=2bbc8a)](https://numpy.org/)
[![](https://img.shields.io/badge/OS-Mac-green?style=f?style=flat-square&logo=macos&logoColor=white)](https://www.apple.com/macos/ventura/)
[![](https://img.shields.io/badge/OS-Windows-green?style=f?style=flat-square&logo=windows&logoColor=white)](https://www.microsoft.com/)
[![](https://img.shields.io/badge/Git_Update-16_Jun_2023-brightgreen?style=f?style=flat-square&logo=github&logoColor=white)](https://github.com/)

# <h3>การทำ Machine Learning (ML)</h3>

**CODE** : <br>

<a href="https://colab.research.google.com/github/HikariJadeEmpire/THNumber_img_classification/blob/main/numberclassifier.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>

<h4>STEP 1</h4>
ในขั้นตอนเริ่มต้น เราก็จะทำการรวบรวม DATA : รูปภาพของตัวเลขไทยที่เขียนด้วยลายมือ :crayon: ( ขนาด 28x28 pixels ) , เพื่อนำไป Train <br>
ซึ่งจะได้ออกมาเป็นตัวเลขละ 70 รูป โดยจะแสดงให้เห็นตัวอย่างของ DATA คร่าวๆ ดังนี้ : <br>
<br>

![output](https://github.com/HikariJadeEmpire/THNumber_img_classification/assets/118663358/42e4d3e4-8038-4e66-bc5d-846cf0556799)

จากนั้นเราจะทำการ Clean DATA :broom: ด้วยวิธีการ <br>
ตัดขอบภาพ >> แปลงเป็นภาพ ขาว-ดำ >> ทำการ Rescale ให้เป็น 28x28 pixels เหมือนตอนเริ่มต้น >> รวบรวม DATA แล้ว Transform ให้เป็น **.CSV** File <br>
  
  **Rescale** EXAMPLE :
  
![Ud-1](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-/assets/118663358/c24388bb-a872-45cf-bf93-63d095a577e9)
  
**.CSV** File EXAMPLE :
  
![Capture](https://github.com/HikariJadeEmpire/THNumber_img_classification/assets/118663358/fd2af6c3-fbc8-4fa7-b6f0-7a3e211567b3)

# <h4>STEP 2</h4>
ขั้นตอนต่อจากนี้ เราจะทำการ **Cross Validation** ด้วยการใช้  [Pycaret :triangular_flag_on_post:](https://pycaret.gitbook.io/docs/) <br>
เพื่อค้นหา Model ที่มี Score โดยเฉลี่ยสูงที่สุด 3-5 อันดับแรก :trophy: แล้วนำไปปรับ ( Tune Model ) เพื่อนำไปใช้ในการ Train & Test ในขั้นตอนสุดท้าย <br>
  
[Pycaret :triangular_flag_on_post:](https://pycaret.gitbook.io/docs/) score :
  
![cap0](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-/assets/118663358/aa3d9c75-a53b-4b92-9723-7f388194c5d9)

  <br>
  
  *NOTE :* ลำดับของ Model อาจมีการเปลี่ยนแปลง เนื่องจากมีการ Re-sampling DATA ในทุกๆครั้งที่ Train
  
  <h4>PLOT : Extra Trees Classifier :deciduous_tree:</h4>
  
  ![output0](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-/assets/118663358/f8e9580a-dd9d-4307-930d-0edcd7bcb94e)
  
  ![output1](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-/assets/118663358/c87ddd5e-273f-4388-b14c-383e2fe047cb)

  
  # <h4>STEP 3</h4>
  Train & Test :books: <br>
  <br>
  
  ใน STEP นี้ เราจะนำ Model ที่มี Score เฉลี่ยดีที่สุด มาใช้ในการ Train & Test :books: โดยอาศัย Setting เดียวกันกับขั้นตอน **Cross Validation** <br>
  ซึ่ง Model ที่ได้มาก็คือ : **Extra Trees Classifier** :deciduous_tree:<br>
  
  ผลลัพธ์ที่ได้จากการ Test : **Extra Trees Classifier Model** :deciduous_tree: ได้ผลลัพธ์ออกมาดังนี้ : <br>
  <br>
  
  ![cap00](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-/assets/118663358/84f1edfd-543d-4f00-adfb-ccf99ae36f12)

  <br>
  จาก Score ด้านบน จะพบว่าคะแนนที่ได้จากการ Test ค่อนข้างดีเยี่ยม โดยจะมีความแม่นยำอยู่ที่ราวๆ 81 % - 100 % <br>
  <br>
  
*NOTE :* Score และ Model อาจมีการเปลี่ยนแปลง เนื่องจากมีการ Re-sampling DATA ในทุกๆครั้งที่ Train <br>

  # 
  Go to top : [Top :compass:](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-#%EF%B8%8F-thnumber_img_classification--without-deep-learning-)
  
  
  
  # <h3>การทำ Application :toolbox::wrench:</h3>
  
  Tools for Developing Applications :<br> 
  
  [![](https://img.shields.io/badge/code-python3.9-green?style=f?style=flat-square&logo=python&logoColor=white&color=2bbc8a)](https://www.python.org/)
  [![](https://img.shields.io/badge/tools-dash-green?style=f?style=flat-square&logo=plotly&logoColor=white&color=2bbc8a)](https://dash.plotly.com/)
  [![](https://img.shields.io/badge/tools-SkLearn-green?style=f?style=flat-square&logo=scikitlearn&logoColor=white&color=2bbc8a)](https://scikit-learn.org/stable/)
  
  
  สำหรับหัวข้อนี้ เราจะอธิบายไปทีละ Page โดยอาศัยวิธีการในการ Clean DATA :broom: และ Train & Test <br>
  แบบเดียวกันกับหัวข้อ : [การทำ Machine Learning (ML)](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-#%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%97%E0%B8%B3-machine-learning-ml) ในระบบหลังบ้าน <br>
  
  ก่อนอื่น เราจะเริ่มด้วยการเปิดใช้งาน App แบบ local โดย ***run*** >> 
  ['home00.py'](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-/blob/main/home00.py) 
  ทั้ง file ด้วย python <br>
  แล้วกดเข้าไปที่ *link* ซึ่งได้จากการ run file ดังกล่าว <br>

หรือ ทำการทดลองใช้งานแบบ Online ผ่าน [ngrok](https://ngrok.com/) <br>
*( วิธีนี้สามารถทดลองใช้งานได้ก็ต่อเมื่อ Host ทำการ Run Original file เอาไว้เท่านั้น )*
  
  # <h4> :one: PAGE 1</h4>
  
  ลักษณะหน้าตาของ *Page 1 : Home ( Importing data )*
  
  <img width="996" alt="Screenshot 2566-06-13 at 19 07 44" src="https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-/assets/118663358/4a5ec0e5-7573-4666-b5ee-69a15d7c62b7">

  <br>
  <br>
  
  สำหรับหน้าแรก จะมีไว้สำหรับ 📥 import **.CSV** file ( ที่ได้จาก [การทำ Machine Learning (ML)](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-#%E0%B8%81%E0%B8%B2%E0%B8%A3%E0%B8%97%E0%B8%B3-machine-learning-ml) ในขั้นตอนของการ Clean DATA :broom: ) เพื่อการนำไปใช้งาน <br>
  เมื่อ Upload หรือ 📥 import เสร็จเรียบร้อยแล้ว ตรงที่เขียนว่า *"No files yet!"* ก็จะเปลี่ยนเป็น *"df_00.csv"* ( เป็นชื่อที่ถูกกำหนดไว้ ไม่ว่าจะ Upload file ชื่อใดๆมาก็ตาม ) <br>
  
  ในกรณีที่ต้องการไปยัง #️⃣ Page อื่นๆ สามารถ *click* 🖱️ ที่แถบด้านบนสุด โดยสามารถเลือกหัวข้อได้ตามที่ต้องการ <br>
  
  # <h4> 2️⃣ PAGE 2</h4>
  
  ลักษณะหน้าตาของ *Page 2 : TRAIN*
  <br>
  
  <img width="914" alt="Screenshot 2566-06-13 at 19 33 06" src="https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-/assets/118663358/33ed2902-ccfc-4248-ab85-17605167b545">

  <br>
  <br>
  สำหรับใน 2️⃣ PAGE 2 จะเป็น #️⃣ Page ที่มีหน้าที่สำหรับการ Train model หรือการทำ 
  Cross Validation
  ซึ่งสามารถเลือก Y ( Target column ) <br>
  และสามารถเลือกจำนวน Training set ( Split ) ได้ <br>
  <br>
  
  จากนั้นก็จะมีช่องสำหรับเลือก Model ที่จะนำไป Train <br>
  โดยผลลัพธ์ที่ได้หลังจากการ *เลือก หรือ ตั้งค่า Model* เสร็จแล้ว ก็จะออกมาเป็น Graph 📈 สำหรับการเปรียบเทียบคะแนนโดยเฉลี่ยของแต่ละ Model <br>
  
  โดยในท้ายที่สุด ก็จะตัดสินใจเลือก Model ที่ :bust_in_silhouette: Users ( ผู้ใช้งาน ) คิดว่าดีที่สุด เพื่อนำไปใช้ในการ Predict ( ทำนาย ) ตัวเลข <br>
  
  # <h4> 3️⃣ PAGE 3</h4>
  
  ลักษณะหน้าตาของ *Page 3 : TEST & PREDICT*
  <br>
  
  <img width="670" alt="Screenshot 2566-06-13 at 21 12 29" src="https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-/assets/118663358/364d3806-377a-45eb-b220-7372bae568f3">

  <br>
  <br>
  สำหรับใน 3️⃣ PAGE 3 จะเป็น #️⃣ Page ที่มีหน้าที่สำหรับการ Test & Predict model <br>
  
  โดยจะมีการปรับ >> ตั้งค่า Model ให้เหมาะสม ตาม Model ที่ได้ทำการเลือกมาแล้วจาก  *Page 2 : TRAIN* 
  <br><br>
  เมื่อตั้งค่าต่างๆเสร็จแล้ว :heavy_check_mark: ก็จะมี Scores ต่างๆ ( Precision, Recall, Accuracy ) แสดงขึ้นมา พร้อมกับ ROC Graph <br>
  ซึ่งเราสามารถใช้ Function บริเวณด้านล่าง ( **IMAGE Prediction** ) ในการ Upload :outbox_tray: รูปภาพของตัวเลขไทย ซึ่งเขียนด้วยลายมือ :crayon: ( ขนาด 28x28 pixels ) ที่เราได้ทำการเขียนขึ้นมาใหม่ ( เช่น เขียนขึ้นโดยโปรแกรม Paint ) ซึ่งจะเป็นตัวเลขอะไรก็ได้ระหว่าง 0-9 <br>
  
  โดยในท้ายที่สุด ก็จะมีการแสดงผลลัพธ์ออกมาที่ Function บริเวณมุมขวาล่าง ( ตัวเลขสีแดง ) พร้อมกับข้อความที่บอกว่า ตัวเลขที่ Upload :outbox_tray: เข้ามาคือตัวเลขอะไร 
  <br>
  
  # 
  Go to top : [Top :compass:](https://github.com/HikariJadeEmpire/THNumber_img_classification-dash_app-#%EF%B8%8F-thnumber_img_classification--without-deep-learning-)
