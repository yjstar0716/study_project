# Generated by Django 3.1.6 on 2021-04-22 08:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0003_ride_weather'),
    ]

    operations = [
        migrations.CreateModel(
            name='Weeks',
            fields=[
                ('day1', models.DateField(primary_key=True, serialize=False)),
                ('week', models.IntegerField(blank=True, null=True)),
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
                'db_table': 'weeks',
                'managed': False,
            },
        ),
    ]