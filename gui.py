import os.path

import PySimpleGUI as sg
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from test import predict
from laneDetection import predict
from HoG import hog


def image_to_data(im):
    """
    Image object to bytes object.
    : Parameters
      im - Image object
    : Return
      bytes object.
    """
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
        print('This is the type pf data ',type(data))
    return data


sg.theme('DarkBrown')
width, height = size = 640, 480  # Scale image

layout = [[sg.Text('Choose your image'), sg.Input(key='File'), sg.FileBrowse(key='-file-')],
          [sg.Button('Submit'), sg.Button('Classify'), sg.Button('Detect'), sg.Button('Exit')],
          [sg.Image(key='-image-')],
          [sg.Text(key='-text-')]]

window = sg.Window('Final Project', layout)

while True:
    event, values = window.read()
    print(event, values)

    if event in (None, 'Exit'):
        break

    if event == 'Submit':
        # Update the "output" text element
        # to be the value of "input" element
        print(values['-file-'])
        if os.path.exists(values['-file-']):
            image = values['-file-']
            print('Image: ', image)
            # window['-image-'].update(image)
            # window['File'].update(image)
            try:
                im = Image.open(image)
            except UnidentifiedImageError:
                print("Cannot identify image file !")
                continue
            w, h = im.size
            scale = min(width / w, height / h, 1)
            if scale != 1:
                im = im.resize((int(w * scale), int(h * scale)))
            data = image_to_data(im)
            window['-image-'].update(data=data, size=size)

    if event == 'Classify':
        print(values)
        window['-text-'].update(predict(values['-file-']))

    if event == 'Detect':
        print(values)
        window['-text-'].update(hog(values['-file-']))
window.close()
