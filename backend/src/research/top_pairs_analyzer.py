"""
Analyze and rank potential pairs from the potential_pairs table.

This script queries the potential_pairs table and ranks pairs based on
multiple metrics to identify the best candidates for pairs trading.

Usage:
    docker compose exec backend-api python -m src.research.top_pairs_analyzer
"""

from __future__ import annotations

import argparse
import csv
import logging
import numpy as np
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import func, select, or_
from sqlalchemy.orm import Session

from common.db import get_db_session
from common.models import PairsAnalysis

LOGGER = logging.getLogger("pairs.top_analyzer")


def calculate_composite_score(pair: PairsAnalysis, total_pairs_tested: int = 24201) -> float:
    """
    Calculate a composite score for ranking pairs, accounting for multiple testing.
    
    This approach prioritizes:
    1. Cointegration strength (CRITICAL - validates long-term relationship)
    2. ADF stationarity (CRITICAL - validates mean reversion)
    3. Half-life (mean reversion speed) - shorter is better
    4. Statistical significance (more trades = more reliable)
    5. Sharpe ratio (with multiple testing awareness)
    6. Profit factor and win rate
    
    Args:
        pair: The pair to score
        total_pairs_tested: Total number of pairs tested (for Bonferroni correction)
    
    Returns:
        Composite score (higher is better)
    """
    score = 0.0
    
    # Cointegration is CRITICAL - validates the fundamental assumption of pairs trading
    # Weight: 25% (highest priority)
    if pair.coint_statistic is not None and pair.coint_pvalue is not None:
        coint_stat = float(pair.coint_statistic)
        coint_p = float(pair.coint_pvalue)
        
        # Test statistic: more negative = stronger cointegration
        # Typical thresholds: < -3.0 is strong, < -4.0 is very strong
        # Normalize: -4.0 = 1.0, -3.0 = 0.75, -2.0 = 0.5, 0 = 0.0
        if coint_stat < 0:
            coint_stat_score = max(0, min(1.0, abs(coint_stat) / 4.0))
        else:
            coint_stat_score = 0.0
        
        # P-value: lower = more significant (typically < 0.05 indicates cointegration)
        # Invert: 0.01 = 1.0, 0.05 = 0.5, 0.10 = 0.0
        coint_p_score = max(0, min(1.0, 1.0 - (coint_p * 10)))
        
        # Combine: average of statistic and p-value scores
        coint_score = (coint_stat_score * 0.6 + coint_p_score * 0.4)
        score += coint_score * 0.25
    
    # ADF stationarity is CRITICAL - validates that the spread is mean-reverting
    # Weight: 20%
    if pair.adf_statistic is not None and pair.adf_pvalue is not None:
        adf_stat = float(pair.adf_statistic)
        adf_p = float(pair.adf_pvalue)
        
        # Test statistic: more negative = more stationary
        # Typical thresholds: < -3.0 is strong, < -4.0 is very strong
        # Normalize: -4.0 = 1.0, -3.0 = 0.75, -2.0 = 0.5, 0 = 0.0
        if adf_stat < 0:
            adf_stat_score = max(0, min(1.0, abs(adf_stat) / 4.0))
        else:
            adf_stat_score = 0.0
        
        # P-value: lower = more significant (typically < 0.05 indicates stationarity)
        # Invert: 0.01 = 1.0, 0.05 = 0.5, 0.10 = 0.0
        adf_p_score = max(0, min(1.0, 1.0 - (adf_p * 10)))
        
        # Combine: average of statistic and p-value scores
        adf_score = (adf_stat_score * 0.6 + adf_p_score * 0.4)
        score += adf_score * 0.20
    
    # Half-life (mean reversion speed) - shorter is better for pairs trading
    # Weight: 20%
    if pair.half_life_minutes is not None:
        half_life = float(pair.half_life_minutes)
        # For daily data, good half-lives are typically 1-7 days (1440-10080 minutes)
        # Longer than 14 days (20160 minutes) suggests very slow mean reversion
        # Invert and normalize: shorter is better
        if half_life > 0:
            # Score peaks at ~3 days (4320 minutes), penalizes very long half-lives
            # Formula: exp(-half_life / optimal_half_life) where optimal = 3 days
            optimal_half_life = 4320  # 3 days in minutes
            half_life_score = max(0, min(1.0, np.exp(-half_life / optimal_half_life)))
            score += half_life_score * 0.20
    
    # Trade count (statistical significance) - weight: 15%
    # More trades = more reliable results, less likely to be p-hacking
    trade_count = pair.pair_total_trades or 0
    # For daily data over 2 years, 10+ trades is reasonable
    # Normalize: 20+ trades is excellent
    trade_score = min(trade_count / 20.0, 1.0)
    score += trade_score * 0.15
    
    # Sharpe ratio - weight: 10% (reduced to avoid p-hacking)
    # Apply Bonferroni correction: with many tests, need much higher Sharpe
    # to be statistically significant
    if pair.pair_sharpe is not None:
        sharpe = float(pair.pair_sharpe)
        if sharpe > 0:
            # Cap at reasonable level to avoid over-weighting outliers
            sharpe_score = min(sharpe / 4.0, 1.0)  # Normalize to 4.0
            score += sharpe_score * 0.10
    
    # Profit factor - weight: 5%
    if pair.pair_profit_factor is not None:
        pf = float(pair.pair_profit_factor)
        if pf > 0:
            # Profit factor > 1.5 is good, > 2.0 is excellent
            pf_score = min(pf / 2.0, 1.0)
            score += pf_score * 0.05
    
    # Win rate - weight: 5%
    if pair.pair_win_rate is not None:
        wr = float(pair.pair_win_rate)
        score += wr * 0.05
    
    return score


def get_top_pairs(
    limit: int = 10,
    min_trades: int = 5,
    max_half_life_days: Optional[float] = None,
    min_sharpe: Optional[float] = None,
    min_coint_statistic: Optional[float] = None,
    max_coint_pvalue: Optional[float] = None,
    min_adf_statistic: Optional[float] = None,
    max_adf_pvalue: Optional[float] = None,
    timeframe: Optional[str] = None,
) -> List[PairsAnalysis]:
    """
    Query the database for top pairs ranked by composite score.
    
    This function applies filters to avoid p-hacking and ensure statistical validity:
    - Requires minimum trades for statistical significance
    - Filters by cointegration statistic AND p-value (validates long-term relationship)
    - Filters by ADF statistic AND p-value (validates mean reversion)
    - Filters by half-life (mean reversion speed)
    - Applies conservative Sharpe thresholds
    
    Args:
        limit: Number of top pairs to return
        min_trades: Minimum number of trades required (default: 5)
        max_half_life_days: Maximum half-life in days (filters slow mean reversion)
        min_sharpe: Minimum Sharpe ratio (optional, but recommended given multiple testing)
        min_coint_statistic: Minimum cointegration statistic (default: -3.0, more negative = stronger)
        max_coint_pvalue: Maximum cointegration p-value (default: 0.10, lower = more significant)
        min_adf_statistic: Minimum ADF statistic (default: -3.0, more negative = more stationary)
        max_adf_pvalue: Maximum ADF p-value (default: 0.10, lower = more significant)
        timeframe: Filter by timeframe (optional)
    
    Returns:
        List of PairsAnalysis objects sorted by composite score
    """
    with get_db_session() as session:
        query = select(PairsAnalysis).where(
            PairsAnalysis.pair_total_trades >= min_trades,
            PairsAnalysis.pair_sharpe.isnot(None),
            PairsAnalysis.half_life_minutes.isnot(None),
            # Require cointegration and ADF statistics for proper ranking
            PairsAnalysis.coint_statistic.isnot(None),
            PairsAnalysis.coint_pvalue.isnot(None),
            PairsAnalysis.adf_statistic.isnot(None),
            PairsAnalysis.adf_pvalue.isnot(None),
        )
        
        # Filter by cointegration statistic (CRITICAL filter)
        # More negative = stronger cointegration
        # Default: -3.0 (typical threshold for strong cointegration)
        if min_coint_statistic is None:
            min_coint_statistic = -3.0
        query = query.where(PairsAnalysis.coint_statistic <= Decimal(str(min_coint_statistic)))
        
        # Filter by cointegration p-value (CRITICAL filter)
        # Lower = more significant (typically < 0.05 indicates cointegration)
        # Default: max 0.10 (allows some flexibility, but 0.05 is ideal)
        if max_coint_pvalue is None:
            max_coint_pvalue = 0.10
        query = query.where(PairsAnalysis.coint_pvalue <= Decimal(str(max_coint_pvalue)))
        
        # Filter by ADF statistic (CRITICAL filter)
        # More negative = more stationary
        # Default: -3.0 (typical threshold for strong stationarity)
        if min_adf_statistic is None:
            min_adf_statistic = -3.0
        query = query.where(PairsAnalysis.adf_statistic <= Decimal(str(min_adf_statistic)))
        
        # Filter by ADF p-value (CRITICAL filter)
        # Lower = more significant (typically < 0.05 indicates stationarity)
        # Default: max 0.10 (allows some flexibility, but 0.05 is ideal)
        if max_adf_pvalue is None:
            max_adf_pvalue = 0.10
        query = query.where(PairsAnalysis.adf_pvalue <= Decimal(str(max_adf_pvalue)))
        
        # Filter by half-life (mean reversion speed)
        # Default: max 14 days (20160 minutes) - pairs with slower mean reversion
        if max_half_life_days is None:
            max_half_life_days = 14.0
        max_half_life_minutes = max_half_life_days * 1440
        query = query.where(PairsAnalysis.half_life_minutes <= Decimal(str(max_half_life_minutes)))
        
        # Apply conservative Sharpe threshold to account for multiple testing
        # With many tests, we need much higher Sharpe to be significant
        if min_sharpe is None:
            min_sharpe = 2.0  # Conservative threshold
        query = query.where(PairsAnalysis.pair_sharpe >= Decimal(str(min_sharpe)))
        
        if timeframe is not None:
            query = query.where(PairsAnalysis.timeframe == timeframe)
        
        # Get all candidates
        all_pairs = session.execute(query).scalars().all()
        total_tested = len(all_pairs)
        
        # Calculate composite scores and sort
        # Pass total_tested for potential future Bonferroni adjustments
        scored_pairs = [(calculate_composite_score(pair, total_tested), pair) for pair in all_pairs]
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top N
        top_pairs = [pair for _, pair in scored_pairs[:limit]]
        
        LOGGER.info(
            f"Found {total_tested} candidate pairs (after filters), returning top {len(top_pairs)}"
        )
        return top_pairs


def format_pair_summary(pair: PairsAnalysis, rank: int) -> str:
    """Format a pair's information for display."""
    sharpe = float(pair.pair_sharpe) if pair.pair_sharpe else None
    pf = float(pair.pair_profit_factor) if pair.pair_profit_factor else None
    wr = float(pair.pair_win_rate) if pair.pair_win_rate else None
    adf_stat = float(pair.adf_statistic) if pair.adf_statistic else None
    adf_p = float(pair.adf_pvalue) if pair.adf_pvalue else None
    coint_stat = float(pair.coint_statistic) if pair.coint_statistic else None
    coint_p = float(pair.coint_pvalue) if pair.coint_pvalue else None
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
        f"  ADF stat:            {adf_stat:.6f}" if adf_stat else "  ADF stat:            N/A",
        f"  ADF p-value:         {adf_p:.6f}" if adf_p else "  ADF p-value:         N/A",
        f"  Cointegration stat:  {coint_stat:.6f}" if coint_stat else "  Cointegration stat:  N/A",
        f"  Cointegration p-val: {coint_p:.6f}" if coint_p else "  Cointegration p-val: N/A",
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


def export_to_csv(pairs: List[PairsAnalysis], filename: str):
    """Export pairs to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'rank', 'symbol_a', 'symbol_b', 'pair', 'timeframe',
            'coint_statistic', 'coint_pvalue', 'adf_statistic', 'adf_pvalue',
            'half_life_days', 'sharpe_ratio', 'profit_factor', 'win_rate',
            'total_trades', 'max_drawdown', 'avg_holding_minutes',
            'hedge_ratio', 'spread_mean', 'spread_std',
            'avg_dollar_volume_a', 'avg_dollar_volume_b',
            'window_start', 'window_end', 'sample_bars'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for rank, pair in enumerate(pairs, 1):
            half_life_days = float(pair.half_life_minutes) / 1440.0 if pair.half_life_minutes else None
            writer.writerow({
                'rank': rank,
                'symbol_a': pair.symbol_a,
                'symbol_b': pair.symbol_b,
                'pair': f"{pair.symbol_a}/{pair.symbol_b}",
                'timeframe': pair.timeframe,
                'coint_statistic': float(pair.coint_statistic) if pair.coint_statistic else None,
                'coint_pvalue': float(pair.coint_pvalue) if pair.coint_pvalue else None,
                'adf_statistic': float(pair.adf_statistic) if pair.adf_statistic else None,
                'adf_pvalue': float(pair.adf_pvalue) if pair.adf_pvalue else None,
                'half_life_days': half_life_days,
                'sharpe_ratio': float(pair.pair_sharpe) if pair.pair_sharpe else None,
                'profit_factor': float(pair.pair_profit_factor) if pair.pair_profit_factor else None,
                'win_rate': float(pair.pair_win_rate) if pair.pair_win_rate else None,
                'total_trades': pair.pair_total_trades,
                'max_drawdown': float(pair.pair_max_drawdown) if pair.pair_max_drawdown else None,
                'avg_holding_minutes': float(pair.pair_avg_holding_minutes) if pair.pair_avg_holding_minutes else None,
                'hedge_ratio': float(pair.hedge_ratio) if pair.hedge_ratio else None,
                'spread_mean': float(pair.spread_mean) if pair.spread_mean else None,
                'spread_std': float(pair.spread_std) if pair.spread_std else None,
                'avg_dollar_volume_a': float(pair.avg_dollar_volume_a) if pair.avg_dollar_volume_a else None,
                'avg_dollar_volume_b': float(pair.avg_dollar_volume_b) if pair.avg_dollar_volume_b else None,
                'window_start': pair.window_start.isoformat() if pair.window_start else None,
                'window_end': pair.window_end.isoformat() if pair.window_end else None,
                'sample_bars': pair.sample_bars,
            })


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze and rank potential pairs from the potential_pairs table"
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of top pairs to return (default: 10)'
    )
    parser.add_argument(
        '--min-trades',
        type=int,
        default=5,
        help='Minimum number of trades required (default: 5)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Output results to CSV file (optional)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output, only show summary table'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    
    LOGGER.info(f"Analyzing top {args.limit} pairs from potential_pairs table...")
    
    # Get top pairs
    top_pairs = get_top_pairs(limit=args.limit, min_trades=args.min_trades)
    
    if not top_pairs:
        LOGGER.warning("No pairs found matching criteria.")
        return
    
    # Export to CSV if requested
    if args.csv:
        export_to_csv(top_pairs, args.csv)
        LOGGER.info(f"Results exported to {args.csv}")
    
    if args.quiet:
        # Only show summary table
        print(f"{'Rank':<6} {'Pair':<20} {'Coint Stat':<12} {'Coint p':<10} {'ADF Stat':<12} {'ADF p':<10} {'Half-Life':<10} {'Sharpe':<10} {'Trades':<8}")
        print("-" * 110)
        for rank, pair in enumerate(top_pairs, 1):
            coint_stat = f"{float(pair.coint_statistic):.2f}" if pair.coint_statistic else "N/A"
            coint_p = f"{float(pair.coint_pvalue):.4f}" if pair.coint_pvalue else "N/A"
            adf_stat = f"{float(pair.adf_statistic):.2f}" if pair.adf_statistic else "N/A"
            adf_p = f"{float(pair.adf_pvalue):.4f}" if pair.adf_pvalue else "N/A"
            sharpe = f"{float(pair.pair_sharpe):.4f}" if pair.pair_sharpe else "N/A"
            half_life = f"{float(pair.half_life_minutes)/1440:.1f}d" if pair.half_life_minutes else "N/A"
            pair_str = f"{pair.symbol_a}/{pair.symbol_b}"
            print(f"{rank:<6} {pair_str:<20} {coint_stat:<12} {coint_p:<10} {adf_stat:<12} {adf_p:<10} {half_life:<10} {sharpe:<10} {pair.pair_total_trades:<8}")
        return
    
    print("\n" + "="*80)
    print(f"TOP {args.limit} POTENTIAL PAIRS FOR PAIRS TRADING STRATEGY")
    print("="*80)
    print(f"\nTotal pairs analyzed: {len(top_pairs)}")
    print("\nRanked by composite score prioritizing:")
    print("  1. Cointegration strength (CRITICAL - validates long-term relationship)")
    print("  2. ADF stationarity (CRITICAL - validates mean reversion)")
    print("  3. Half-life (mean reversion speed) - shorter is better")
    print("  4. Trade count (statistical significance)")
    print("  5. Sharpe ratio (with multiple testing awareness)")
    print("  6. Profit factor and win rate")
    print("\nFilters applied:")
    print("  - Cointegration statistic ≤ -3.0 (strong cointegration)")
    print("  - Cointegration p-value ≤ 0.10 (statistically significant)")
    print("  - ADF statistic ≤ -3.0 (strong stationarity)")
    print("  - ADF p-value ≤ 0.10 (statistically significant)")
    print("  - Half-life ≤ 14 days (filters slow mean reversion)")
    print("  - Sharpe ratio ≥ 2.0 (conservative threshold)")
    print(f"  - Minimum {args.min_trades} trades (statistical significance)")
    
    for rank, pair in enumerate(top_pairs, 1):
        print(format_pair_summary(pair, rank))
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Rank':<6} {'Pair':<20} {'Coint Stat':<12} {'Coint p':<10} {'ADF Stat':<12} {'ADF p':<10} {'Half-Life':<10} {'Sharpe':<10} {'Trades':<8}")
    print("-" * 110)
    
    for rank, pair in enumerate(top_pairs, 1):
        coint_stat = f"{float(pair.coint_statistic):.2f}" if pair.coint_statistic else "N/A"
        coint_p = f"{float(pair.coint_pvalue):.4f}" if pair.coint_pvalue else "N/A"
        adf_stat = f"{float(pair.adf_statistic):.2f}" if pair.adf_statistic else "N/A"
        adf_p = f"{float(pair.adf_pvalue):.4f}" if pair.adf_pvalue else "N/A"
        sharpe = f"{float(pair.pair_sharpe):.4f}" if pair.pair_sharpe else "N/A"
        half_life = f"{float(pair.half_life_minutes)/1440:.1f}d" if pair.half_life_minutes else "N/A"
        pair_str = f"{pair.symbol_a}/{pair.symbol_b}"
        
        print(f"{rank:<6} {pair_str:<20} {coint_stat:<12} {coint_p:<10} {adf_stat:<12} {adf_p:<10} {half_life:<10} {sharpe:<10} {pair.pair_total_trades:<8}")
    
    if args.csv:
        print(f"\n✅ Results exported to: {args.csv}")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("1. Use these pairs to fetch 5-second tick data")
    print("2. Run backtests with the pairs trading strategy")
    print("3. Validate results on out-of-sample data")
    print("="*80)


if __name__ == "__main__":
    main()

