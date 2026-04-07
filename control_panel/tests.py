from django.test import TestCase

from control_panel.model_registry import get_model_choices, read_model_bytes
from control_panel.models import TrainingJob


class TrainingJobStorageTests(TestCase):
    def test_retention_limit_keeps_only_latest_weight_blobs(self):
        for index in range(TrainingJob.MAX_STORED_MODELS + 2):
            TrainingJob.objects.create(
                name=f"Model {index}",
                status='COMPLETED',
                model_weights=f"weights-{index}".encode(),
            )

        retained_jobs = TrainingJob.objects.filter(model_weights__isnull=False).order_by('id')
        self.assertEqual(retained_jobs.count(), TrainingJob.MAX_STORED_MODELS)
        self.assertFalse(TrainingJob.objects.get(name="Model 0").model_weights)
        self.assertFalse(TrainingJob.objects.get(name="Model 1").model_weights)

    def test_database_model_reference_reads_binary_payload(self):
        job = TrainingJob.objects.create(
            name="Render Model",
            status='COMPLETED',
            model_weights=b'binary-model-payload',
        )

        payload = read_model_bytes(f"db:{job.id}")
        self.assertEqual(payload, b'binary-model-payload')

    def test_model_choices_include_database_models(self):
        job = TrainingJob.objects.create(
            name="Named Model",
            status='COMPLETED',
            model_weights=b'abc123',
        )

        choices = get_model_choices(include_disk=False, include_database=True)
        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0]['value'], f"db:{job.id}")
        self.assertIn("Named Model", choices[0]['label'])
