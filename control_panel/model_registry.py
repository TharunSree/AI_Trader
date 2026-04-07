from pathlib import Path

from .models import TrainingJob


def normalize_model_reference(reference: str) -> str:
    return (reference or '').strip()


def is_database_model_reference(reference: str) -> bool:
    return normalize_model_reference(reference).startswith('db:')


def get_database_model(reference: str):
    normalized = normalize_model_reference(reference)
    if not is_database_model_reference(normalized):
        raise ValueError(f"Unsupported database model reference: {reference}")

    _, _, raw_id = normalized.partition(':')
    if not raw_id.isdigit():
        raise ValueError(f"Invalid database model id: {reference}")

    return TrainingJob.objects.filter(id=int(raw_id)).first()


def read_model_bytes(reference: str) -> bytes:
    normalized = normalize_model_reference(reference)
    if is_database_model_reference(normalized):
        job = get_database_model(normalized)
        if not job or not job.model_weights:
            raise FileNotFoundError(f"Database model {reference} has no stored weights.")
        return bytes(job.model_weights)

    model_path = Path(normalized)
    if not model_path.exists():
        candidate = Path("saved_models") / normalized
        if candidate.exists():
            model_path = candidate

    if not model_path.exists():
        raise FileNotFoundError(f"Weight file {reference} not found.")

    return model_path.read_bytes()


def get_model_label(reference: str) -> str:
    normalized = normalize_model_reference(reference)
    if is_database_model_reference(normalized):
        job = get_database_model(normalized)
        if not job:
            return normalized
        return f"{job.name} ({normalized})"

    return Path(normalized).name


def get_model_choices(include_disk: bool = True, include_database: bool = True):
    choices = []

    if include_database:
        for job in TrainingJob.objects.filter(model_weights__isnull=False).order_by('-id'):
            choices.append({
                'value': job.model_reference,
                'label': f"{job.name} ({job.model_reference})",
                'source': 'database',
            })

    if include_disk:
        model_dir = Path("saved_models")
        if model_dir.exists():
            for model_file in sorted(model_dir.glob("*.pth"), reverse=True):
                choices.append({
                    'value': model_file.name,
                    'label': model_file.name,
                    'source': 'disk',
                })

    return choices
