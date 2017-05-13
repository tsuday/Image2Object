from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.init),
    url(r'^predict$', views.predict),
]