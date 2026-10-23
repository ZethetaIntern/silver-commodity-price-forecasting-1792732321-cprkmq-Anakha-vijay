"""Deterministic educational trading arena logic; no live trading connectivity."""
SCENARIOS = {
    'normal_market': {'shock_pct': 1.0, 'volatility_pct': 18, 'liquidity': 1.0, 'description': 'Ordinary session with modest directional movement.'},
    'crisis_selloff': {'shock_pct': -8.0, 'volatility_pct': 65, 'liquidity': 0.55, 'description': 'Risk-off shock, widening spreads and rapid de-risking.'},
    'flash_crash': {'shock_pct': -12.0, 'volatility_pct': 95, 'liquidity': 0.25, 'description': 'Sudden air pocket followed by partial rebound; limit exposure.'},
    'supply_disruption': {'shock_pct': 6.0, 'volatility_pct': 48, 'liquidity': 0.7, 'description': 'Mine or refining disruption tightens expected supply.'},
    'sentiment_squeeze': {'shock_pct': 10.0, 'volatility_pct': 80, 'liquidity': 0.45, 'description': 'Crowded positioning and retail attention create squeeze risk.'},
    'manipulation_attempt': {'shock_pct': 7.0, 'volatility_pct': 90, 'liquidity': 0.35, 'description': 'Suspicious order/volume spike; require anomaly review and avoid assuming the move is fundamental.'},
}

def score_round(equity, exposure_pct, position, shock_pct, fee_bps=5, slippage_bps=10):
    """Return a simple simulated P&L and cost estimate. Position: long, short, cash."""
    signed = {'long': 1, 'short': -1, 'cash': 0}[position.lower()]
    notional = equity * exposure_pct / 100
    gross = notional * signed * shock_pct / 100
    costs = notional * (fee_bps + slippage_bps) / 10000
    pnl = gross - costs
    return {'notional': notional, 'gross_pnl': gross, 'estimated_costs': costs, 'net_pnl': pnl, 'ending_equity': equity + pnl}
