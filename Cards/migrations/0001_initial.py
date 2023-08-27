# Generated by Django 4.2.4 on 2023-08-16 20:14

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='FormDataModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('START_X', models.IntegerField()),
                ('STOP_X', models.IntegerField()),
                ('START_Y', models.IntegerField()),
                ('STOP_Y', models.IntegerField()),
                ('CSV_LIMIT_RECORDS', models.IntegerField()),
                ('OUTPUT_CSV', models.CharField(max_length=255)),
            ],
        ),
    ]