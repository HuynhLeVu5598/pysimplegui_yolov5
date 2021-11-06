import PySimpleGUI as sg
from PySimpleGUI.PySimpleGUI import WIN_CLOSED, Checkbox
import cv2
import numpy as np 
from PIL import Image,ImageTk
import torch
import io 
import os

# CLASSES1 =  [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
#             'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
#             'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
#             'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
#             'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#             'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
#             'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
#             'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
#             'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
#             'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
#             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
#             'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
#             'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
#             'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]

# CLASSES2 = [ 'bt' ]

# CLASSES3 = ['bactruc', 'traybactruc', 'tron']

#CLASSES = CLASSES1

#model =torch.hub.load('./yolov5','custom', path= 'D:/test/app1/yolov5s.pt', source='local',force_reload =True)
CLASSES = []

list_spin = [i for i in range(101)]
file_img = [("JPEG (*.jpg)",("*jpg","*.png"))]

file_weights = [('Weights (*.pt)', ('*.pt'))]

sg.theme('Dark')
layout = [
    [sg.Text('Detection', size =(47,1),justification='center' ,font= ('Helvetica',30),text_color='red' ,relief= sg.RELIEF_SUNKEN)],
    [
        sg.Image(filename='', size=(640,480),key='image',background_color='black'),
        sg.Frame('Parameter',[
            #[sg.Text('Weights', size=(12,1), font=('Helvetica',15),text_color= 'yellow'), sg.InputOptionMenu(('something','bactruc','vonho'),size=(20,20),default_value='something',key='weights')],
            [sg.Text('Weights', size=(12,1), font=('Helvetica',15),text_color='yellow'), sg.Input(size=(19,1), font=('Helvetica',12), key='file_weights',readonly= True, text_color='navy') ,sg.FileBrowse(file_types= file_weights, size=(10,1), font=('Helvetica',10))],
            [sg.Text('Size', size=(12,1),font=('Helvetica',15), text_color='yellow'),sg.InputCombo((416,512,608,896,1024,1280,1408,1536),size=(23,20),default_value=416,key='imgsz')],
            [sg.Text('Confidence',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,100),orientation='h',size=(20,20),default_value=25, key= 'conf_thres')],
            #[sg.Text('IOU',size=(12,1),font=('Helvetica',15),text_color='yellow'), sg.Slider(range=(1,100),orientation='h', size=(20,20),default_value=45, key='iou_thres')],
            [sg.Text('Max detection', size=(12,1), font=('Helvetica',15), text_color='yellow'), sg.Spin(values=list_spin, initial_value=1, size=(23,20),key='max_det')],
            [sg.Text('Classes', size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Listbox(values=CLASSES,size=(23,4), text_color= 'aqua',select_mode= sg.LISTBOX_SELECT_MODE_MULTIPLE, key='classes')],
            [sg.Text('Result',size=(12,1),font=('Helvetica',15),text_color='yellow'),sg.InputText('',size=(16,20),justification='center',font=('Helvetica',15),text_color='red',readonly=True, key='result')]
        ],font=('Helvetica',20),title_color='orange')
    ],
    [   sg.Frame('',
        [
            [sg.Button('Webcam', size=(10,1),  font=('Helvetica',15),disabled=True),
            sg.Button('Stop', size=(10,1), font=('Helvetica',15))]
        ]),
        sg.Frame('',
        [
            [sg.Button('Change',size=(10,1), font=('Helvetica',15)),
            sg.Button('Exit', size=(10,1), font=('Helvetica',15))]
        ])
    ],
    [sg.Frame('',[
        [
            sg.Text('Image File',size =(10,1), font=('Helvetica',15)),
            sg.Input(size=(20,1),font=('Helvetica',12),key='filename',readonly= True,text_color='navy'),
            sg.FileBrowse(file_types=file_img, size=(10,1),font=('Helvetica',15)),
            sg.Button('Show', size=(10,1), font=('Helvetica',15)),
            sg.Button('Detect', size=(10,1), font=('Helvetica',15),disabled=True)
        ]
    ])]
]   


window = sg.Window('Window', layout, location=(0,0),grab_anywhere=True)

#model = torch.hub.load('ultralytics/yolov5','yolov5s')
#model =torch.hub.load('yolov5/','custom', path= 'yolov5s.pt', source='local',force_reload =True)

# model =torch.hub.load('./yolov5','custom', path= 'D:/test/app1/yolov5s.pt', source='local',force_reload =True)
# CLASSES = model.names

cap = cv2.VideoCapture(0)
recording = False

while True:
    event, values = window.read(timeout=20)

    resulting = False
  
    values_classes=[]
    for i in values['classes']:
        values_classes.append(CLASSES.index(i))
 
    if event == 'Exit' or event == sg.WIN_CLOSED:
        break

    elif event == 'Webcam':
        recording = True

    elif event == 'Stop':
        recording = False 
        img = np.full((480,640),0)
        imgbytes = cv2.imencode('.png', img)[1].tobytes()
        window['image'].update(data= imgbytes)
        window['result'].update('')

    if event == 'Change':
        # if values['weights'] == 'something':
        #     model =torch.hub.load('./yolov5','custom', path='yolov5s.pt', source='local',force_reload =True)
        #     CLASSES = CLASSES1
        #     window['classes'].update(values=CLASSES)


        # elif values['weights'] == 'bactruc':
        #     model =torch.hub.load('./yolov5','custom', path='best_bt.pt', source='local',force_reload =True)
        #     CLASSES = CLASSES2
        #     window['classes'].update(values=CLASSES)
        
        # elif values['weights'] == 'vonho':
        #     model =torch.hub.load('./yolov5','custom',path='bestv5s-1280.pt',source='local', force_reload=True)
        #     CLASSES = CLASSES3 
        #     window['classes'].update(values=CLASSES)
        # print(values['file_weights'])
        # print(type(values['file_weights']))
        model= torch.hub.load('./yolov5','custom',path=values['file_weights'],source='local',force_reload=True)
        CLASSES= model.names
        window['classes'].update(values=CLASSES)
        if CLASSES is not None:
            window['Webcam'].update(disabled= False)
            window['Detect'].update(disabled= False)

    if recording:
        ret, frame = cap.read()
        result = model(frame,size= values['imgsz'],conf = values['conf_thres']/100, max_det= values['max_det'], classes= values_classes)
        for i, pred in enumerate(result.pred):
            if pred.shape[0]:
                for *box,cof,clas in reversed(pred):
                    if result.names[int(clas.tolist())] == 'person':
                        window['result'].update('NG')
            else:
                window['result'].update('OK')
        #window['result'].update('') 

        show = np.squeeze(result.render()) 
        imgbytes = cv2.imencode('.png',show)[1].tobytes()
        window['image'].update(data=imgbytes)

    if event == 'Show':
        window['result'].update('')
        file_name = values['filename']
        if os.path.exists(file_name):
            image = Image.open(file_name)
            image.thumbnail((640, 480))
            photo_img = ImageTk.PhotoImage(image)
            window['image'].update(data=photo_img)           

    if event == 'Detect':
        file_name = values['filename']
        if os.path.exists(file_name):
            #img = Image.open(file_name)
            #img.thumbnail((720,1280))
            #img = np.array(img)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # image = Image.open(file_name)
            # image.thumbnail((640, 480))
            # bio = io.BytesIO()
            # image.save(bio, format="PNG")
            # window['image'].update(data=bio.getvalue())

            #img = file_name
            #result = model(img,size= values['imgsz'],conf = values['conf_thres']/100, iou= values['iou_thres']/100, max_det= values['max_det'], classes= values_classes)
            #img_result = np.squeeze(result.render())
            #imgbytes = cv2.imencode('.png',img_result)[1].tobytes()
            #window['image'].update(data=bio.getvalue())

            #image = Image.open(file_name)
            #image.thumbnail((400, 400))
            #photo_img = ImageTk.PhotoImage(image)
            #window['image'].update(data=photo_img)


            # image = Image.open(file_name)
            # image.thumbnail((400, 400))
            # img = image
            # result = model(img,size= values['imgsz'],conf = values['conf_thres']/100, iou= values['iou_thres']/100, max_det= values['max_det'], classes= values_classes)
            # img_result = np.squeeze(result.render())
            # imgbytes = cv2.imencode('.png',img_result)[1].tobytes()
            # window['image'].update(data= imgbytes)
            window['result'].update('')
            image = Image.open(file_name)
            image.thumbnail((640, 480))
            img = image
            result = model(img,size= values['imgsz'],conf = values['conf_thres']/100, max_det= values['max_det'], classes= values_classes)
            for i, pred in enumerate(result.pred):
                if pred.shape[0]:
                    for *box,cof,clas in reversed(pred):               
                        if result.names[int(clas.tolist())] == 'traybactruc' or result.names[int(clas.tolist())] =='person':
                            window['result'].update('NG')
                            resulting = True
                        else:
                            if resulting == False:
                                window['result'].update('OK')
                
            #window['result'].update('') 
            
            img_result = np.squeeze(result.render())
            img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
            imgbytes = cv2.imencode('.png',img_result)[1].tobytes()
            window['image'].update(data= imgbytes)

window.close() 

#pyinstaller --onefile app.py yolov5/hubconf.py yolov5/models/common.py yolov5/models/experimental.py yolov5/models/yolo.py yolov5/utils/augmentations.py yolov5/utils/autoanchor.py yolov5/utils/datasets.py yolov5/utils/downloads.py yolov5/utils/general.py yolov5/utils/metrics.py yolov5/utils/plots.py yolov5/utils/torch_utils.py
#pyinstaller --onedir --windowed app.py yolov5/hubconf.py yolov5/models/common.py yolov5/models/experimental.py yolov5/models/yolo.py yolov5/utils/augmentations.py yolov5/utils/autoanchor.py yolov5/utils/datasets.py yolov5/utils/downloads.py yolov5/utils/general.py yolov5/utils/metrics.py yolov5/utils/plots.py yolov5/utils/torch_utils.py                       