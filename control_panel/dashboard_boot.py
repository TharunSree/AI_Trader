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
    recent_trades=None,
):
    market_open = bool(clock_data and clock_data.get('is_open'))
    top_positions = sorted(
        [
            {
                'symbol': getattr(position, 'symbol', ''),
                'quantity': getattr(position, 'qty', getattr(position, 'quantity', '0')),
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

    # Format positions for JS table rendering
    formatted_positions = []
    for p in (positions or []):
        formatted_positions.append({
            'symbol': getattr(p, 'symbol', ''),
            'qty': float(getattr(p, 'qty', 0)),
            'market_value': float(getattr(p, 'market_value', 0)),
        })

    # Format recent_trades for JS table rendering
    formatted_trades = []
    for t in (recent_trades or []):
        formatted_trades.append({
            'timestamp': t.timestamp.strftime('%H:%M:%S') if t.timestamp else '',
            'symbol': t.symbol,
            'action': t.action,
            'price': float(t.price or 0),
            'notional_value': float(t.notional_value or 0),
            'trader_id': t.trader_id,
        })

    return {
        'positions': formatted_positions,
        'recent_trades': formatted_trades,
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
                'value': 'ONLINE' if getattr(trader, 'status', '') == 'RUNNING' else 'STANDBY',
                'badge': getattr(trader, 'status', 'STOPPED') if trader else 'STOPPED',
                'meta': trader_model_label,
            },
            {
                'id': 'training-pulse',
                'kind': 'progress',
                'title': 'AI Training Pulse',
                'value': training_mode,
                'progress': int(training_progress),
                'badge': 'META ACTIVE' if active_meta and active_meta.status == 'RUNNING' else 'ROI: 0%' if active_training and active_training.status == 'RUNNING' else 'AWAITING BATCH',
            },
            {
                'id': 'equity-timeline',
                'kind': 'chart',
                'title': 'Equity Timeline',
                'points': [18, 24, 21, 35, 31, 46, 42, 58],
            },
            {
                'id': 'reward-optimization',
                'kind': 'chart',
                'title': 'Neural Reward Optimization',
                'points': [14, 18, 22, 25, 29, 33, 36, 41],
            },
            {
                'id': 'live-positions',
                'kind': 'list',
                'title': 'Live Positions',
                'items': top_positions or [{'symbol': 'SYS', 'market_value': 0}],
            },
        ],
    }
