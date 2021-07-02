from django.urls import path
from django.conf.urls import url
from home import views

app_name = ''
urlpatterns = [
 path('', views.home, name='home'),
 path('analysis', views.analysis, name='analysis'),
 path('rnn_analysis', views.rnn_analysis, name='rnn_analysis'),
 path('ann_analysis', views.ann_analysis, name='ann_analysis'),
 url(r'^please', views.please, name='please'),
 path('staff', views.staff, name='staff'),
 url(r'^idea', views.idea, name='idea'),
 url(r'^idea_call', views.idea_call, name='idea_call'),
 url(r'^korea_call', views.korea_call, name='korea_call'),
 url(r'^rnn_call', views.rnn_call, name='rnn_call'),
 url(r'^ann_call', views.ann_call, name='ann_call'),
 path('pv/', views.pv, name='pv'),
 path('blog/<int:curr_page>/', views.blog, name='blog'),
 path('blog/1/', views.blog, name='blogpower'),
 path('detail/<int:num1>/', views.detail, name='detail'),
]