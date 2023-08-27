from Cards import views
from django.urls import path

urlpatterns = [
    path('', views.index, name = 'index'),
    path('video', views.video_feed,name = 'video_feed'),
    path('json_data', views.json_data, name = 'json_data'),
    path('peak_detection', views.peak_detection, name = 'peak_detection'),
    path('stream_video', views.stream_video, name = 'stream_video'),
    path('stream_peakvideo', views.stream_peakvideo, name = 'stream_peakvideo'),
    path('stream_all', views.stream_all, name = 'stream_all'),
    path('Peak_jsondata', views.Peak_jsondata, name = 'Peak_jsondata'),
    path('streamfromweb', views.streamfromweb, name = 'streamfromweb'),
    path('peakfromweb', views.peakfromweb, name = 'peakfromweb'),
    
]

