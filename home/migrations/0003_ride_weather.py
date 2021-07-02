# Generated by Django 3.1.6 on 2021-04-19 02:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0002_auto_20210405_1258'),
    ]

    operations = [
        migrations.CreateModel(
            name='Ride',
            fields=[
                ('days', models.DateField(primary_key=True, serialize=False)),
                ('weeks', models.CharField(blank=True, max_length=50, null=True)),
                ('gangwon', models.IntegerField(blank=True, null=True)),
                ('geonggi', models.IntegerField(blank=True, null=True)),
                ('gyeongnam', models.IntegerField(blank=True, null=True)),
                ('gyeongbook', models.IntegerField(blank=True, null=True)),
                ('gwangju', models.IntegerField(blank=True, null=True)),
                ('daegu', models.IntegerField(blank=True, null=True)),
                ('daejeon', models.IntegerField(blank=True, null=True)),
                ('busan', models.IntegerField(blank=True, null=True)),
                ('seoul', models.IntegerField(blank=True, null=True)),
                ('sejong', models.IntegerField(blank=True, null=True)),
                ('incheon', models.IntegerField(blank=True, null=True)),
                ('jeonnam', models.IntegerField(blank=True, null=True)),
                ('jeonbook', models.IntegerField(blank=True, null=True)),
                ('jeju', models.IntegerField(blank=True, null=True)),
                ('chungnam', models.IntegerField(blank=True, null=True)),
                ('chungbook', models.IntegerField(blank=True, null=True)),
            ],
            options={
                'db_table': 'ride',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Weather',
            fields=[
                ('days', models.DateField(primary_key=True, serialize=False)),
                ('gangwon', models.CharField(blank=True, max_length=50, null=True)),
                ('geonggi', models.CharField(blank=True, max_length=50, null=True)),
                ('gyeongnam', models.CharField(blank=True, max_length=50, null=True)),
                ('gyeongbook', models.CharField(blank=True, max_length=50, null=True)),
                ('gwangju', models.CharField(blank=True, max_length=50, null=True)),
                ('daegu', models.CharField(blank=True, max_length=50, null=True)),
                ('daejeon', models.CharField(blank=True, max_length=50, null=True)),
                ('busan', models.CharField(blank=True, max_length=50, null=True)),
                ('seoul', models.CharField(blank=True, max_length=50, null=True)),
                ('sejong', models.CharField(blank=True, max_length=50, null=True)),
                ('incheon', models.CharField(blank=True, max_length=50, null=True)),
                ('jeonnam', models.CharField(blank=True, max_length=50, null=True)),
                ('jeonbook', models.CharField(blank=True, max_length=50, null=True)),
                ('jeju', models.CharField(blank=True, max_length=50, null=True)),
                ('chungnam', models.CharField(blank=True, max_length=50, null=True)),
                ('chungbook', models.CharField(blank=True, max_length=50, null=True)),
            ],
            options={
                'db_table': 'weather',
                'managed': False,
            },
        ),
    ]
