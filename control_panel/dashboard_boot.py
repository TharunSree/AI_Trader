from .model_registry import get_model_label


def build_dashboard_boot_payload(
    *,
    live_equity,
    buying_power,
    positions,
    clock_data,
    active_meta,
    active_training,
    trader,
):
    market_open = bool(clock_data and clock_data.get('is_open'))
    top_positions = sorted(
        [
            {
                'symbol': getattr(position, 'symbol', ''),
                'market_value': float(getattr(position, 'market_value', 0) or 0),
            }
            for position in (positions or [])
        ],
        key=lambda item: item['market_value'],
        reverse=True,
    )[:3]

    training_mode = 'STANDBY'
    training_progress = 0
    if active_meta:
        training_mode = f"META {active_meta.status}"
        training_progress = active_meta.progress or 0
    elif active_training:
        training_mode = f"STD {active_training.status}"
        training_progress = active_training.progress or 0

    trader_ref = getattr(trader, 'model_file', '') or ''
    trader_model_label = get_model_label(trader_ref) if trader_ref else 'No model linked'

    return {
        'header': {
            'equity': float(live_equity or 0),
            'buying_power': float(buying_power or 0),
            'market_status': 'OPEN' if market_open else 'CLOSED',
            'trader_status': getattr(trader, 'status', 'OFFLINE') if trader else 'OFFLINE',
            'trader_model_label': trader_model_label,
            'training_mode': training_mode,
            'training_progress': int(training_progress),
        },
        'components': [
            {
                'id': 'account-equity',
                'kind': 'metric',
                'title': 'Account Equity',
                'value': f"${float(live_equity or 0):,.2f}",
                'meta': f"BP: ${float(buying_power or 0):,.2f}",
            },
            {
                'id': 'engine-status',
                'kind': 'status',
                'title': 'Engine Status',
                'value': getattr(trader, 'status', 'STANDBY') if trader else 'STANDBY',
                'meta': trader_model_label,
            },
            {
                'id': 'training-pulse',
                'kind': 'progress',
                'title': 'AI Training Pulse',
                'value': training_mode,
                'progress': int(training_progress),
            },
            {
                'id': 'equity-timeline',
                'kind': 'chart',
                'title': 'Equity Timeline',
                'points': [18, 24, 21, 35, 31, 46, 42, 58],
            },
            {
                'id': 'live-positions',
                'kind': 'list',
                'title': 'Live Positions',
                'items': top_positions or [{'symbol': 'SYS', 'market_value': 0}],
            },
        ],
    }
