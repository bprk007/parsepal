from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import layoutparser as lp
import cv2
import os

#from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    if request.method=='POST':
        # data=CustomData(
        #     gender=request.form.get('gender'),
        #     race_ethnicity=request.form.get('ethnicity'),
        #     parental_level_of_education=request.form.get('parental_level_of_education'),
        #     lunch=request.form.get('lunch'),
        #     test_preparation_course=request.form.get('test_preparation_course'),
        #     reading_score=float(request.form.get('writing_score')),
        #     writing_score=float(request.form.get('reading_score'))

        # )

        # pred_df=data.get_data_as_data_frame()
        # print(pred_df)
        # print("Before Prediction")

        # predict_pipeline=PredictPipeline()
        # print("Mid Prediction")
        # results=predict_pipeline.predict(pred_df)
        # print("after Prediction")

        


        file  = request.files['img']
        if not file:
            return ({"error: no image uploaded"}, 400)
        
        if file:
            # Save the uploaded image
            image_path = os.path.join(application.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            img = image_path
            # return render_template('home.html', img=img)
        
        imagee = cv2.imread(img)
        model = lp.Detectron2LayoutModel(
        config_path = os.path.join("artifacts","config.yaml"),
        model_path=os.path.join("artifacts","model_0039999.pth"),
        extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # <-- Only output high accuracy preds
        )
        layout = model.detect(imagee)
        lp.draw_box(imagee, layout)
        text_blocks = lp.Layout([b for b in layout if b.type==0])
        figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])
        h, w = imagee.shape[:2]

        left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(imagee)

        left_blocks = text_blocks.filter_by(left_interval, center=True)
        left_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)
        # The b.coordinates[1] corresponds to the y coordinate of the region
        # sort based on that can simulate the top-to-bottom reading order 
        right_blocks = lp.Layout([b for b in text_blocks if b not in left_blocks])
        right_blocks.sort(key = lambda b:b.coordinates[1], inplace=True)

        # And finally combine the two lists and add the index
        text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])

        lp.draw_box(imagee, text_blocks,
            box_width=3, 
            show_element_id=True)
        
        import layoutparser.ocr as ocr
        ocr_agent = ocr.TesseractAgent(languages="eng+hin")

        for block in text_blocks:
            segment_image = (block
                            .pad(left=5, right=5, top=5, bottom=5)
                            .crop_image(imagee))
            lp.draw_box(segment_image, text_blocks,
                    box_width=3, 
                    show_element_id=True)
                # add padding in each image segment can help
                # improve robustness 
                
            text = ocr_agent.detect(segment_image)
            block.set(text=text, inplace=True)

        lp.draw_box(segment_image, text_blocks,
            box_width=3, 
            show_element_id=True)
        
        for txt in text_blocks.get_texts():
            print(txt, end='\n---\n')

        segment_image = (text_blocks[7]
                       .pad(left=5, right=5, top=5, bottom=5)
                       .crop_image(imagee))
        lp.draw_box(segment_image, text_blocks,
                box_width=3, 
                show_element_id=True)
        
        print("text is ",txt)
        return render_template('home.html', results = txt)
    


if __name__=="__main__":
    app.run(host="0.0.0.0")        

