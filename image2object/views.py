from django.shortcuts import render

import numpy as np
import tensorflow as tf

# import django modules for CSRF
from django.views.decorators.csrf import csrf_protect
from django.template.context_processors import csrf
from django.shortcuts import render_to_response

# import django modules for Http handling
from django.http import HttpResponse
from django.http import Http404

import re
import json
import base64

# import AutoEncoder program
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/libraries")
from AutoEncoder import AutoEncoder


# PIL
from PIL import Image
from io import BytesIO

# Regex to extract base64 image data
imageUrlPattern = re.compile('drawnImage=data:image/png;base64,(.*)$')


# Initialize AutoEncoder
print("Start initializing AutoEncoder...")
autoEncoder = AutoEncoder("", 1, False)
autoEncoder.saver.restore(autoEncoder.sess, os.path.dirname(__file__) + "/tensorflow_session/s-36000")
print("End initializing AutoEncoder...")


# response to request to the top page
@csrf_protect
def init(request):

    # CSRF
    c = {}
    c.update(csrf(request))

    return render_to_response('app.html', c);
    #return render_to_response(request, 'app.html', c);

# response predicted image to the user drawn image
@csrf_protect
def predict(request):

    if request.method != 'POST':
        raise Http404

    # Convert bytes(request.body) to string
    drawnImage = request.body.decode();

    # Remove base64 header "data:image/png;base64,"
    drawnImageStr = imageUrlPattern.match(drawnImage).group(1)

    # Open with Pillow
    drawnImageImg = Image.open(BytesIO(base64.b64decode(drawnImageStr)))
    
    # Convert to numpy array and reshape
    # drawnImage shape : (512, 512, 4)
    drawnImageArray = np.asarray(drawnImageImg)
    
    # retrieve alpha value from RGBA
    drawnImageAlphaArray = 255 - drawnImageArray[:, :, 3]

    # reshape to (1, 512*512)
    drawnImageInput = drawnImageAlphaArray.reshape((1, AutoEncoder.nPixels))

    # out shape : (1, 512, 512, 1)
    out, x_input = autoEncoder.sess.run([autoEncoder.output, autoEncoder.x_image], feed_dict={autoEncoder.x:drawnImageInput, autoEncoder.keep_prob:1.0})


    # arrange np array as the same shape as input "drawnImage"
    predictedImageArray = drawnImageArray
    predictedImageArray.flags.writeable = True

    output = 255 - out[0, :, :, 0]
    np.clip(output, 0, 255, out=output)
    predictedImageArray[:, :, 3] = output

    # Convert numpy array to Pillow image
    predictedImage = Image.fromarray(np.uint8(predictedImageArray))

    # Convert Pillow image to base64
    bufferdata = BytesIO()
    predictedImage.save(bufferdata, format="PNG")
    bufferdata.seek(0)
    predictedImageBytes = base64.b64encode(bufferdata.read())
    predictedImageStr = predictedImageBytes.decode("ascii")

    # Return JSON format
    response = json.dumps({ 'predictedImage' : "data:image/png;base64," + predictedImageStr })

    return HttpResponse(response, content_type="text/javascript")

