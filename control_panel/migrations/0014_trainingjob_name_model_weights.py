from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('control_panel', '0013_remove_trainingjob_num_episodes_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingjob',
            name='model_weights',
            field=models.BinaryField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trainingjob',
            name='name',
            field=models.CharField(default='Untitled Model', max_length=100),
        ),
    ]
