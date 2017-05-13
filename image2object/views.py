from django.shortcuts import render

from django.views.decorators.csrf import csrf_protect
from django.template.context_processors import csrf
from django.shortcuts import render_to_response

from django.http import HttpResponse
from django.http import Http404

import json
import re
import base64

# Regex to extract base64 image data
imageUrlPattern = re.compile('drawnImage=data:image/png;base64,(.*)$')

# response to request to the top page
@csrf_protect
def init(request):

    # CSRF
    c = {}
    c.update(csrf(request))

    return render_to_response('app.html', c);
    #return render_to_response(request, 'app.html', c);
    #return render(request, 'app.html', {})

# response predicted image to the user drawn image
@csrf_protect
def predict(request):

    if request.method != 'POST':
        raise Http404

    # convert bytes(request.body) to string by "decode"
    drawnImage = request.body.decode();

    drawnImage = imageUrlPattern.match(drawnImage).group(1)

    #print(drawnImage)

     # JSON format
    response = json.dumps({ 'predictedImage' : "data:image/png;base64," + drawnImage })

    return HttpResponse(response, content_type="text/javascript")

