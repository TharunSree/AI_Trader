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


from django.contrib.auth.models import User
from django.urls import reverse
from django.core.cache import cache
from unittest.mock import patch
import json

class DashboardViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="testpassword")

    def test_dashboard_redirects_if_anonymous(self):
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)

    def test_dashboard_authenticated_success(self):
        self.client.force_login(self.user)
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        
        # Verify dashboard_boot_payload context variable exists and is a valid JSON string
        self.assertIn('dashboard_boot_payload', response.context)
        payload_str = response.context['dashboard_boot_payload']
        payload = json.loads(payload_str)
        
        self.assertIn('positions', payload)
        self.assertIn('recent_trades', payload)
        self.assertIn('header', payload)

    @patch('control_panel.views.psutil')
    def test_telemetry_cpu_fallback(self, mock_psutil):
        # Test case where cache is empty and psutil returns 0.0
        if not mock_psutil:
            self.skipTest("psutil not available")
            
        mock_psutil.cpu_percent.return_value = 0.0
        cache.delete("system_telemetry_cpu")
        
        from control_panel.views import _get_memory_snapshot
        snapshot = _get_memory_snapshot()
        self.assertIsNotNone(snapshot)
        self.assertIn('cpu_percent', snapshot)
        # Fallback should result in a value between 0.5 and 2.5
        self.assertTrue(0.5 <= snapshot['cpu_percent'] <= 2.5)

    @patch('control_panel.views.psutil')
    def test_telemetry_cpu_cached(self, mock_psutil):
        if not mock_psutil:
            self.skipTest("psutil not available")
            
        cache.set("system_telemetry_cpu", 45.2, timeout=10)
        from control_panel.views import _get_memory_snapshot
        snapshot = _get_memory_snapshot()
        self.assertEqual(snapshot['cpu_percent'], 45.2)
        cache.delete("system_telemetry_cpu")

    def test_download_report_view_anonymous_redirect(self):
        response = self.client.get(reverse('neural_download_report'))
        self.assertEqual(response.status_code, 302)

    def test_download_report_view_mock_fallback(self):
        self.client.force_login(self.user)
        # Verify that it loads successfully even with no PaperTrader objects in database (since it falls back to mock)
        response = self.client.get(reverse('neural_download_report'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Simulated Paper Trader (Demo Mode)")
        self.assertContains(response, "best_model.pth")
        
        # Test markdown download
        response_md = self.client.get(reverse('neural_download_report') + '?format=markdown')
        self.assertEqual(response_md.status_code, 200)
        self.assertEqual(response_md['Content-Type'], 'text/markdown')
        self.assertIn(b'# AI Trader - Neural Cortex & Diagnostics Report', response_md.content)

    def test_record_decision_api_creates_log(self):
        self.client.force_login(self.user)
        # First register a PaperTrader bot
        from control_panel.models import PaperTrader, OnlineLearningLog
        trader = PaperTrader.objects.create(
            status="RUNNING",
            model_file="best_model.pth"
        )
        
        payload = {
            'trader_id': trader.id,
            'price_trend': 0.5,
            'sentiment': -0.2,
            'volatility': 0.15,
            'spread': 0.05,
            'action_val': 0.65
        }
        # Post to endpoint
        response = self.client.post(
            reverse('record_decision_api'),
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        resp_data = json.loads(response.content)
        self.assertEqual(resp_data['status'], 'success')
        
        # Verify event was recorded in database
        self.assertEqual(OnlineLearningLog.objects.filter(trader=trader, event_type='DECISION').count(), 1)

