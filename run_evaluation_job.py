import os
import django
import argparse
from pathlib import Path

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "trader_project.settings")
django.setup()

from control_panel.models import EvaluationJob
from src.sessions.evaluation_session import EvaluationSession

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, required=True)
    args = parser.parse_args()

    job = EvaluationJob.objects.filter(id=args.job_id).first()
    if not job:
        print(f"Error: EvaluationJob {args.job_id} not found.")
        return

    job.status = 'RUNNING'
    job.save(update_fields=['status'])

    try:
        # Run evaluation
        results = EvaluationSession({
            'model_file': job.model_file,
            'start_date': str(job.start_date),
            'end_date': str(job.end_date),
        }).run()

        job.results = results
        job.status = 'COMPLETED'
        job.save(update_fields=['results', 'status'])
        print(f"Evaluation {job.id} completed successfully.")
        
    except Exception as e:
        job.status = 'FAILED'
        job.error_message = str(e)
        job.save(update_fields=['status', 'error_message'])
        print(f"Evaluation {job.id} failed: {e}")

if __name__ == "__main__":
    main()
