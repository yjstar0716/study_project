# Generated by Django 3.1.6 on 2021-03-31 07:14

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Day',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pub_date', models.DateTimeField(verbose_name='date published')),
                ('day_text', models.CharField(max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='Weather',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('weather_text', models.CharField(max_length=200)),
                ('votes', models.IntegerField(default=0)),
                ('day', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='home.day')),
            ],
        ),
    ]
