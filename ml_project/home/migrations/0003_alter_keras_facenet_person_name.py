# Generated by Django 4.1.1 on 2022-11-06 11:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0002_keras_facenet_image_keras_facenet_model_name_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='keras_facenet',
            name='person_name',
            field=models.CharField(max_length=200),
        ),
    ]
