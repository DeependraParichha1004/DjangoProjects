# Generated by Django 4.1.1 on 2022-11-06 11:49

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0003_alter_keras_facenet_person_name'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='keras_facenet',
            name='probability',
        ),
    ]
