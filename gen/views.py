from django.shortcuts import render
import urllib
import numpy as np
import tensorflow as tf

# from PIL import Image

from .apps import FeatureExtModelConfig,CaptionModelConfig,TokenizerConfig
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser
#import cloudinary

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.sequence import pad_sequences
import cloudinary
import cv2

def idx_to_word(integer , tokenizer):
    for word , index in tokenizer.word_index.items():
        if index == integer:
            return word

    return None


def predict_caption(model , image , tokenizer , max_length):

    in_text = 'startseq'

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]

        sequence = pad_sequences([sequence] , max_length)

        y = model.predict([image , sequence] , verbose=0)

        y = np.argmax(y)

        word = idx_to_word(y , tokenizer)

        if word is None:
            break

        in_text = in_text + " " + word

        if word == "endseq":
            break

    return in_text

class UploadView(APIView):
    parser_classes=(
        MultiPartParser,
        JSONParser,
    )
    @staticmethod
    def post(request):
        tokenizer=TokenizerConfig.tokenzizer
        feature_ext=FeatureExtModelConfig.feature_extractor
        model=CaptionModelConfig.model

        file=request.data.get('file')
        upload_data=cloudinary.uploader.upload(file)
        img=upload_data['url']
        req=urllib.request.urlopen(img)

        arr=np.asarray(bytearray(req.read()),dtype=np.uint8)
        image=cv2.imdecode(arr,-1)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,(224,224))
        # image=np.array(image)/255
        image=np.expand_dims(image,axis=0)
        image = preprocess_input(image)
        feature = feature_ext.predict(image , verbose=0)
        
        y_pred = predict_caption(model , feature , tokenizer , 35)
        output_list = y_pred.split()
        output = ''
        for i in range(1,len(output_list)-1):
            output = output + " " + output_list[i]

        return Response(
            {
                "status":"success",
                "data":upload_data,
                "url":img,
                "caption":output,
            },
            status=201
        )
