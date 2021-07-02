# # from django.shortcuts import render, get_object_or_404
# # from django.http import HttpResponseRedirect
# # from django.urls import reverse
#
# from django.views.generic.base import TemplateView
#
# class HomeView(TemplateView):
#
#     template_name = 'home.html'
#     def get_context_data(self, **kwargs):# 오버라이딩, **붙으면 key형태로 온값을 받음(추가할 데이터를 매개변수로 전달)
#         context = super(HomeView, self).get_context_data(**kwargs)
#         return context