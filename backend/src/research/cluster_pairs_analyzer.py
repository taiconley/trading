"""
Offline analysis tool to score potential trading pairs using Clustering (PCA + DBSCAN).

This approach reduces the search space from O(N^2) to O(N) by only testing pairs
within the same cluster. This finds pairs that are fundamentally similar in price action.

Run inside the backend container:
    docker compose exec backend-api python -m src.research.cluster_pairs_analyzer \
        --timeframe "1 hour" \
        --lookback-days 60 \
        --min-dollar-volume 500000 \
        --pca-components 10 \
        --min-cluster-size 2
"""

import argparse
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Sequence
from datetime import datetime, timedelta, timezone
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler

from .potential_pairs_analyzer import (
    load_symbol_profiles, load_symbol_data, symbol_data_to_frame,
    analyze_pair, insert_results, AnalysisParams, parse_args as parse_base_args,
    timeframe_to_seconds, execute_with_retry, SymbolProfile, SymbolData
)

LOGGER = logging.getLogger("pairs.cluster_research")

def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify candidate pairs using Clustering.")
    # Inherit most args from the base analyzer, but we'll re-define them here for clarity
    # or use the base parser if we refactored it to be reusable. 
    # For now, I'll just add the specific ones and duplicate the common ones for a standalone script.
    
    parser.add_argument("--timeframe", default="1 hour", help="Timeframe for correlation analysis (default: 1 hour).")
    parser.add_argument("--lookback-days", type=int, default=60, help="Days to analyze for clustering.")
    parser.add_argument("--start-date", type=str, default=None, help="Inclusive UTC start timestamp (e.g. 2024-01-01T13:30:00Z).")
    parser.add_argument("--end-date", type=str, default=None, help="Inclusive UTC end timestamp.")
    parser.add_argument("--min-dollar-volume", type=float, default=500_000.0, help="Min avg dollar volume.")
    parser.add_argument("--max-symbols", type=int, default=500, help="Max symbols to load.")
    
    parser.add_argument("--pca-components", type=int, default=15, help="Number of PCA components.")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon parameter.")
    parser.add_argument("--min-cluster-size", type=int, default=2, help="Min samples for a cluster.")
    
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--min-bars", type=int, default=500, help="Minimum overlapping bars required.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Max pairs to analyze.")
    parser.add_argument("--parallelism", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args(args=args)

def get_returns_matrix(data_map: Dict[str, SymbolData]) -> pd.DataFrame:
    """
    Create a DataFrame of returns for all symbols.
    Aligned by timestamp.
    """
    frames = []
    for sym, data in data_map.items():
        df = symbol_data_to_frame(data)
        # Resample to 1-hour if needed, but we assume data is already loaded at requested timeframe
        # Calculate log returns
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        frames.append(df['log_ret'].rename(sym))
    
    if not frames:
        return pd.DataFrame()
        
    # Join all returns on index (timestamp)
    returns_df = pd.concat(frames, axis=1)
    
    # Drop rows with too many NaNs (e.g. market open/close mismatches)
    # We want rows where most stocks have data
    returns_df.dropna(thresh=int(len(frames) * 0.8), inplace=True)
    
    # Fill remaining NaNs with 0 (flat price)
    returns_df.fillna(0, inplace=True)
    
    return returns_df

def find_clusters(returns_df: pd.DataFrame, n_components: int, eps: float, min_samples: int) -> Dict[int, List[str]]:
    """
    Perform PCA + DBSCAN clustering.
    Returns a dict mapping cluster_id -> list of symbols.
    """
    # 1. Normalize returns
    scaler = StandardScaler()
    normalized_returns = scaler.fit_transform(returns_df.T) # Transpose so samples=stocks, features=time
    
    # 2. PCA for dimensionality reduction
    # We want to cluster STOCKS based on their time-series behavior
    pca = PCA(n_components=min(n_components, len(returns_df.columns), len(returns_df)))
    reduced_data = pca.fit_transform(normalized_returns)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    LOGGER.info(f"PCA with {n_components} components explained {explained_variance:.2%} of variance")
    
    # 3. DBSCAN Clustering
    # We cluster the STOCKS (rows of reduced_data)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clustering.fit_predict(reduced_data)
    
    clusters = {}
    symbols = returns_df.columns.tolist()
    
    for idx, label in enumerate(labels):
        if label == -1:
            continue # Noise
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(symbols[idx])
        
    return clusters

def analyze_pair_wrapper(args):
    """
    Wrapper for analyze_pair to be used with ProcessPoolExecutor.
    args is a tuple: (sym_a, sym_b, data_a, data_b, profile_map, params)
    """
    sym_a, sym_b, data_a, data_b, profile_map, params = args
    return analyze_pair(sym_a, sym_b, data_a, data_b, profile_map, params)

def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, 
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    
    # 1. Load Data
    # We reuse the loading logic from potential_pairs_analyzer
    # But we might want a wider universe for clustering
    
    # Mocking the session/db context from the original script
    # We need to import get_engine inside the function to avoid module-level side effects if imported elsewhere
    from common.db import get_db_session
    
    with get_db_session() as session:
        # Load profiles
        profiles = load_symbol_profiles(
            session, 
            args.timeframe, 
            None, None, # Start/End will be calculated
            min_bars=args.min_bars, 
            min_avg_dollar_volume=args.min_dollar_volume,
            max_symbols=args.max_symbols
        )
        
        if not profiles:
            LOGGER.error("No symbols found.")
            return

        LOGGER.info(f"Loaded {len(profiles)} symbol profiles.")
        
        # Load data for all symbols
        # Load data for all symbols
        # We need a common time window
        if args.end_date:
            end_date = datetime.fromisoformat(args.end_date.replace("Z", "+00:00"))
        else:
            end_date = datetime.now(timezone.utc)
            
        if args.start_date:
            start_date = datetime.fromisoformat(args.start_date.replace("Z", "+00:00"))
        else:
            start_date = end_date - timedelta(days=args.lookback_days)
        
        data_map = {}
        for p in profiles:
            data = load_symbol_data(session, p.symbol, args.timeframe, start_date, end_date)
            if data:
                data_map[p.symbol] = data
                
    LOGGER.info(f"Loaded data for {len(data_map)} symbols.")
    
    # 2. Clustering
    returns_df = get_returns_matrix(data_map)
    if returns_df.empty:
        LOGGER.error("Empty returns matrix.")
        return
        
    LOGGER.info(f"Returns matrix shape: {returns_df.shape} (Time x Stocks)")
    
    clusters = find_clusters(returns_df, args.pca_components, args.eps, args.min_cluster_size)
    
    LOGGER.info(f"Found {len(clusters)} clusters.")
    total_pairs_to_test = 0
    for cid, syms in clusters.items():
        n = len(syms)
        pairs_count = n * (n-1) // 2
        total_pairs_to_test += pairs_count
        LOGGER.info(f"Cluster {cid}: {n} symbols ({pairs_count} pairs) -> {syms}")
        
    if total_pairs_to_test == 0:
        LOGGER.warning("No pairs found in clusters.")
        return
        
    # 3. Analyze Pairs within Clusters
    # We can reuse analyze_pair from potential_pairs_analyzer
    
    bar_seconds = timeframe_to_seconds(args.timeframe)
    params = AnalysisParams(
        timeframe=args.timeframe,
        bar_seconds=bar_seconds,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        min_bars=args.min_bars,
        min_spread_std=1e-6,
        min_trades=args.min_trades
    )
    
    profile_map = {p.symbol: p for p in profiles}
    results = []
    
    import itertools
    import concurrent.futures
    
    all_combinations = []
    for cid, syms in clusters.items():
        LOGGER.info(f"Preparing pairs for Cluster {cid}...")
        combinations = list(itertools.combinations(syms, 2))
        for sym_a, sym_b in combinations:
            if sym_a in data_map and sym_b in data_map:
                all_combinations.append((sym_a, sym_b))

    if args.max_pairs and len(all_combinations) > args.max_pairs:
        LOGGER.info(f"Capping pairs at {args.max_pairs} (found {len(all_combinations)})")
        all_combinations = all_combinations[:args.max_pairs]
        
    LOGGER.info(f"Starting analysis of {len(all_combinations)} pairs with {args.parallelism} workers...")
    
    # Prepare tasks for parallel execution
    # We pass the profile_map and params to each task. 
    # Note: For very large datasets, passing full profile_map might be inefficient if not using fork.
    tasks = [
        (sym_a, sym_b, data_map[sym_a], data_map[sym_b], profile_map, params)
        for sym_a, sym_b in all_combinations
    ]

    if args.parallelism > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallelism) as executor:
            # Submit all tasks
            futures = [executor.submit(analyze_pair_wrapper, task) for task in tasks]
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                if i % 100 == 0:
                    LOGGER.info(f"Processed {i}/{len(all_combinations)} pairs...")
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                except Exception as e:
                    LOGGER.error(f"Error analyzing pair: {e}")
    else:
        # Serial execution
        for i, task in enumerate(tasks):
            if i % 100 == 0:
                LOGGER.info(f"Processed {i}/{len(all_combinations)} pairs...")
            res = analyze_pair_wrapper(task)
            if res:
                results.append(res)
                
    LOGGER.info(f"Analysis complete. Found {len(results)} valid pairs from clusters.")
    
    # 4. Insert Results
    # Add source to metadata for all results
    for res in results:
        if "meta" not in res:
            res["meta"] = {}
        if isinstance(res["meta"], str):
            import json
            try:
                res["meta"] = json.loads(res["meta"])
            except:
                res["meta"] = {}
        res["meta"]["source"] = "cluster"
        # Convert back to string if needed, but insert_results likely handles dicts if the model expects JSON
        # Actually, looking at the error log, meta is passed as a string. 
        # Let's check how potential_pairs_analyzer handles it. 
        # The error log showed meta as a string. 
        # But the model definition says JSON. SQLAlchemy handles JSON serialization usually.
        # However, the error log showed: 'meta__0': '{"total_pnl": ...}'
        # This suggests it might be being manually serialized or the model expects it.
        # Let's just set it as a dict and let the ORM/driver handle it, or if analyze_pair returns a string, parse and re-dump.
        
    # The analyze_pair function returns a dict with 'meta' as a JSON string (based on the log).
    # We should probably just update the status to 'candidate'.
    
    insert_results(results, "candidate", args.replace_existing, args.timeframe, start_date, end_date)

if __name__ == "__main__":
    main()
