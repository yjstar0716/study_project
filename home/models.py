from django.db import models


class Post(models.Model):
    postname = models.CharField(max_length=50)
    mainphoto = models.ImageField(blank=True, null=True)
    contents = models.TextField()

    # def __str__(self):
    #     return self.postname


class Ride(models.Model):
    days = models.DateField(primary_key=True)
    weeks = models.CharField(max_length=50, blank=True, null=True)
    gangwon = models.IntegerField(blank=True, null=True)
    geonggi = models.IntegerField(blank=True, null=True)
    gyeongnam = models.IntegerField(blank=True, null=True)
    gyeongbook = models.IntegerField(blank=True, null=True)
    gwangju = models.IntegerField(blank=True, null=True)
    daegu = models.IntegerField(blank=True, null=True)
    daejeon = models.IntegerField(blank=True, null=True)
    busan = models.IntegerField(blank=True, null=True)
    seoul = models.IntegerField(blank=True, null=True)
    sejong = models.IntegerField(blank=True, null=True)
    incheon = models.IntegerField(blank=True, null=True)
    jeonnam = models.IntegerField(blank=True, null=True)
    jeonbook = models.IntegerField(blank=True, null=True)
    jeju = models.IntegerField(blank=True, null=True)
    chungnam = models.IntegerField(blank=True, null=True)
    chungbook = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'ride'


class Weather(models.Model):
    days = models.DateField(primary_key=True)
    gangwon = models.CharField(max_length=50, blank=True, null=True)
    geonggi = models.CharField(max_length=50, blank=True, null=True)
    gyeongnam = models.CharField(max_length=50, blank=True, null=True)
    gyeongbook = models.CharField(max_length=50, blank=True, null=True)
    gwangju = models.CharField(max_length=50, blank=True, null=True)
    daegu = models.CharField(max_length=50, blank=True, null=True)
    daejeon = models.CharField(max_length=50, blank=True, null=True)
    busan = models.CharField(max_length=50, blank=True, null=True)
    seoul = models.CharField(max_length=50, blank=True, null=True)
    sejong = models.CharField(max_length=50, blank=True, null=True)
    incheon = models.CharField(max_length=50, blank=True, null=True)
    jeonnam = models.CharField(max_length=50, blank=True, null=True)
    jeonbook = models.CharField(max_length=50, blank=True, null=True)
    jeju = models.CharField(max_length=50, blank=True, null=True)
    chungnam = models.CharField(max_length=50, blank=True, null=True)
    chungbook = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'weather'


class Weeks(models.Model):
    day1 = models.DateField(primary_key=True)
    week = models.IntegerField(blank=True, null=True)
    gangwon = models.IntegerField(blank=True, null=True)
    geonggi = models.IntegerField(blank=True, null=True)
    gyeongnam = models.IntegerField(blank=True, null=True)
    gyeongbook = models.IntegerField(blank=True, null=True)
    gwangju = models.IntegerField(blank=True, null=True)
    daegu = models.IntegerField(blank=True, null=True)
    daejeon = models.IntegerField(blank=True, null=True)
    busan = models.IntegerField(blank=True, null=True)
    seoul = models.IntegerField(blank=True, null=True)
    sejong = models.IntegerField(blank=True, null=True)
    incheon = models.IntegerField(blank=True, null=True)
    jeonnam = models.IntegerField(blank=True, null=True)
    jeonbook = models.IntegerField(blank=True, null=True)
    jeju = models.IntegerField(blank=True, null=True)
    chungnam = models.IntegerField(blank=True, null=True)
    chungbook = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'weeks'


class Board(models.Model):
    num = models.AutoField(primary_key=True)
    day = models.DateField(blank=True, null=True)
    name = models.CharField(max_length=50, blank=True, null=True)
    title = models.CharField(max_length=50,blank=True, null=True)
    text = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'board'
    #def __str__(self):
    #    return self.title