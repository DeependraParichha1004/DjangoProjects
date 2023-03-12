# Generated by Django 4.1.1 on 2022-10-07 17:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='keras_facenet',
            name='image',
            field=models.ImageField(default=0, upload_to='pics'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='keras_facenet',
            name='model_name',
            field=models.CharField(default=0, max_length=40),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='keras_facenet',
            name='person_name',
            field=models.CharField(default=0, max_length=50),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='keras_facenet',
            name='probability',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
    ]
