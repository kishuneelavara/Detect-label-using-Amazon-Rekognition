import pandas as pd
import boto3
import cv2
from botocore.exceptions import ClientError

credential = pd.read_csv("new_user_credentials.csv")
access_key_id = credential['Access key ID'][0]
secret_access_key = credential['Secret access key'][0]

client = boto3.client('rekognition', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)


def get_label(img,label):
    try:

        filename=img
        img='static/uploads/'+filename
        photo1 = cv2.imread(img)
        imgHeight, imgWidth, channels = photo1.shape

        with open(img, 'rb') as source_image:
            source_bytes = source_image.read()

        response = client.detect_labels(
            Image={
                'Bytes': source_bytes
            },
            MaxLabels=int(label),
            MinConfidence=80)

        res_response={}
        for i in range(len(response['Labels'])):
            label='Label'+str(i)
            res_response[label]={}
            res_response[label]['Name']=response['Labels'][i]['Name']
            res_response[label]['Confidence']=response['Labels'][i]['Confidence']
            if len(response['Labels'][i]['Instances']) == 0:
                res_response[label]['Instances']="Boundary Box Not Available"
            else:
                print(response['Labels'][i]['Instances'])
                noOfBoundingBox = len(response['Labels'][i]["Instances"])
                for j in range(0, noOfBoundingBox):
                    dimensions = (response['Labels'][i]["Instances"][j]["BoundingBox"])
                    # Storing them in variables
                    boxWidth = dimensions['Width']
                    boxHeight = dimensions['Height']
                    boxLeft = dimensions['Left']
                    boxTop = dimensions['Top']
                    # Plotting points of rectangle
                    start_point = (int(boxLeft * imgWidth), int(boxTop * imgHeight))
                    end_point = (int((boxLeft + boxWidth) * imgWidth), int((boxTop + boxHeight) * imgHeight))
                    # Drawing Bounding Box on the coordinates
                    thickness = 2
                    color=(36,255,12)
                    photo1 = cv2.rectangle(photo1, start_point, end_point,color , thickness)
                    cv2.putText(photo1,response['Labels'][i]['Name'] , (int(boxLeft * imgWidth), (int(boxTop * imgHeight))-10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, thickness)

                    # cv2.imshow('Target Image', photo1)
                    # cv2.waitKey(0)
            print(res_response)
        cv2.imwrite("static/result/"+filename,photo1)
        statement="success"
        return response, filename, res_response,statement
    except:
        statement="Something went wrong"
        return None, None, None, statement




    # if __name__ == '__main__':
#     img="image2.jpg"
#     get_label(img)