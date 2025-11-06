"""
Analyze and rank potential pairs from the potential_pairs table.

This script queries the potential_pairs table and ranks pairs based on
multiple metrics to identify the best candidates for pairs trading.

Usage:
    docker compose exec backend-api python -m src.research.top_pairs_analyzer
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import func, select, or_
from sqlalchemy.orm import Session

from common.db import get_db_session
from common.models import PairsAnalysis

LOGGER = logging.getLogger("pairs.top_analyzer")


def calculate_composite_score(pair: PairsAnalysis) -> float:
    """
    Calculate a composite score for ranking pairs.
    
    Higher scores indicate better pairs. Considers:
    - Sharpe ratio (weighted heavily)
    - Profit factor
    - Win rate
    - Cointegration strength (lower p-value is better)
    - Trade count (more trades = more statistical significance)
    """
    score = 0.0
    
    # Sharpe ratio (primary metric) - weight: 40%
    if pair.pair_sharpe is not None:
        sharpe = float(pair.pair_sharpe)
        # Normalize: assume good Sharpe is > 1.0, scale accordingly
        score += max(0, sharpe) * 0.4
    
    # Profit factor - weight: 25%
    if pair.pair_profit_factor is not None:
        pf = float(pair.pair_profit_factor)
        # Normalize: profit factor > 1.5 is good
        score += min(pf / 1.5, 2.0) * 0.25
    
    # Win rate - weight: 15%
    if pair.pair_win_rate is not None:
        wr = float(pair.pair_win_rate)
        score += wr * 0.15
    
    # Cointegration strength - weight: 10%
    # Lower p-value is better (stronger cointegration)
    if pair.coint_pvalue is not None:
        coint_p = float(pair.coint_pvalue)
        # Invert: p-value of 0.01 is better than 0.05
        # Scale: 0.01 -> 1.0, 0.05 -> 0.5, 0.1 -> 0.0
        coint_score = max(0, 1.0 - (coint_p * 10))
        score += coint_score * 0.10
    
    # Trade count (statistical significance) - weight: 10%
    # More trades = more reliable results
    trade_count = pair.pair_total_trades or 0
    # Normalize: 50+ trades is excellent, scale down from there
    trade_score = min(trade_count / 50.0, 1.0)
    score += trade_score * 0.10
    
    return score


def get_top_pairs(
    limit: int = 10,
    min_trades: int = 5,
    min_sharpe: Optional[float] = None,
    timeframe: Optional[str] = None,
) -> List[PairsAnalysis]:
    """
    Query the database for top pairs ranked by composite score.
    
    Args:
        limit: Number of top pairs to return
        min_trades: Minimum number of trades required
        min_sharpe: Minimum Sharpe ratio (optional)
        timeframe: Filter by timeframe (optional)
    
    Returns:
        List of PairsAnalysis objects sorted by composite score
    """
    with get_db_session() as session:
        query = select(PairsAnalysis).where(
            PairsAnalysis.pair_total_trades >= min_trades,
            PairsAnalysis.pair_sharpe.isnot(None),
            PairsAnalysis.pair_profit_factor.isnot(None),
        )
        
        if min_sharpe is not None:
            query = query.where(PairsAnalysis.pair_sharpe >= Decimal(str(min_sharpe)))
        
        if timeframe is not None:
            query = query.where(PairsAnalysis.timeframe == timeframe)
        
        # Get all candidates
        all_pairs = session.execute(query).scalars().all()
        
        # Calculate composite scores and sort
        scored_pairs = [(calculate_composite_score(pair), pair) for pair in all_pairs]
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N
        top_pairs = [pair for _, pair in scored_pairs[:limit]]
        
        LOGGER.info(f"Found {len(all_pairs)} candidate pairs, returning top {len(top_pairs)}")
        return top_pairs


def format_pair_summary(pair: PairsAnalysis, rank: int) -> str:
    """Format a pair's information for display."""
    sharpe = float(pair.pair_sharpe) if pair.pair_sharpe else None
    pf = float(pair.pair_profit_factor) if pair.pair_profit_factor else None
    wr = float(pair.pair_win_rate) if pair.pair_win_rate else None
    coint_p = float(pair.coint_pvalue) if pair.coint_pvalue else None
    adf_p = float(pair.adf_pvalue) if pair.adf_pvalue else None
    half_life = float(pair.half_life_minutes) if pair.half_life_minutes else None
    dd = float(pair.pair_max_drawdown) if pair.pair_max_drawdown else None
    avg_hold = float(pair.pair_avg_holding_minutes) if pair.pair_avg_holding_minutes else None
    hedge_ratio = float(pair.hedge_ratio) if pair.hedge_ratio else None
    
    lines = [
        f"\n{'='*80}",
        f"Rank #{rank}: {pair.symbol_a} / {pair.symbol_b}",
        f"{'='*80}",
        f"Timeframe: {pair.timeframe}",
        f"Analysis Window: {pair.window_start} to {pair.window_end}",
        f"Sample Bars: {pair.sample_bars:,}",
        "",
        "Performance Metrics:",
        f"  Sharpe Ratio:        {sharpe:.4f}" if sharpe else "  Sharpe Ratio:        N/A",
        f"  Profit Factor:       {pf:.4f}" if pf else "  Profit Factor:       N/A",
        f"  Win Rate:            {wr:.1%}" if wr else "  Win Rate:            N/A",
        f"  Total Trades:        {pair.pair_total_trades:,}",
        f"  Max Drawdown:        {dd:.6f}" if dd else "  Max Drawdown:        N/A",
        f"  Avg Holding Time:    {avg_hold:.1f} minutes" if avg_hold else "  Avg Holding Time:    N/A",
        "",
        "Statistical Tests:",
        f"  Cointegration p-val: {coint_p:.6f}" if coint_p else "  Cointegration p-val: N/A",
        f"  ADF p-value:         {adf_p:.6f}" if adf_p else "  ADF p-value:         N/A",
        f"  Half-life:           {half_life:.2f} minutes" if half_life else "  Half-life:           N/A",
        "",
        "Pair Characteristics:",
        f"  Hedge Ratio:         {hedge_ratio:.6f}" if hedge_ratio else "  Hedge Ratio:         N/A",
        f"  Spread Mean:         {float(pair.spread_mean):.8f}" if pair.spread_mean else "  Spread Mean:         N/A",
        f"  Spread Std:          {float(pair.spread_std):.8f}" if pair.spread_std else "  Spread Std:          N/A",
        "",
        "Liquidity:",
        f"  Avg $ Volume A:      ${float(pair.avg_dollar_volume_a):,.2f}",
        f"  Avg $ Volume B:      ${float(pair.avg_dollar_volume_b):,.2f}",
        "",
        "Simulation Parameters:",
        f"  Entry Z-score:       {float(pair.simulated_entry_z):.2f}" if pair.simulated_entry_z else "  Entry Z-score:       N/A",
        f"  Exit Z-score:        {float(pair.simulated_exit_z):.2f}" if pair.simulated_exit_z else "  Exit Z-score:        N/A",
    ]
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    
    LOGGER.info("Analyzing top pairs from potential_pairs table...")
    
    # Get top 10 pairs
    top_pairs = get_top_pairs(limit=10, min_trades=5)
    
    if not top_pairs:
        LOGGER.warning("No pairs found matching criteria.")
        return
    
    print("\n" + "="*80)
    print("TOP 10 POTENTIAL PAIRS FOR PAIRS TRADING STRATEGY")
    print("="*80)
    print(f"\nTotal pairs analyzed: {len(top_pairs)}")
    print("\nRanked by composite score (Sharpe, Profit Factor, Win Rate, Cointegration, Trade Count)")
    
    for rank, pair in enumerate(top_pairs, 1):
        print(format_pair_summary(pair, rank))
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Rank':<6} {'Pair':<25} {'Sharpe':<10} {'Profit Factor':<15} {'Win Rate':<12} {'Trades':<10} {'Coint p-val':<12}")
    print("-" * 80)
    
    for rank, pair in enumerate(top_pairs, 1):
        sharpe = f"{float(pair.pair_sharpe):.4f}" if pair.pair_sharpe else "N/A"
        pf = f"{float(pair.pair_profit_factor):.4f}" if pair.pair_profit_factor else "N/A"
        wr = f"{float(pair.pair_win_rate):.1%}" if pair.pair_win_rate else "N/A"
        coint_p = f"{float(pair.coint_pvalue):.6f}" if pair.coint_pvalue else "N/A"
        pair_str = f"{pair.symbol_a}/{pair.symbol_b}"
        
        print(f"{rank:<6} {pair_str:<25} {sharpe:<10} {pf:<15} {wr:<12} {pair.pair_total_trades:<10} {coint_p:<12}")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("1. Use these pairs to fetch 5-second tick data")
    print("2. Run backtests with the pairs trading strategy")
    print("3. Validate results on out-of-sample data")
    print("="*80)


if __name__ == "__main__":
    main()

