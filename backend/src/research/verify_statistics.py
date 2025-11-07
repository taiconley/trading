"""Quick script to verify ADF and cointegration statistics are being captured."""
from common.db import get_db_session
from common.models import PairsAnalysis
from sqlalchemy import select, func

with get_db_session() as session:
    # Check statistics for the test timeframe
    total = session.execute(
        select(func.count(PairsAnalysis.id))
        .where(PairsAnalysis.timeframe == "1 day")
    ).scalar()
    
    with_adf_stat = session.execute(
        select(func.count(PairsAnalysis.id))
        .where(
            PairsAnalysis.timeframe == "1 day",
            PairsAnalysis.adf_statistic.isnot(None)
        )
    ).scalar()
    
    with_coint_stat = session.execute(
        select(func.count(PairsAnalysis.id))
        .where(
            PairsAnalysis.timeframe == "1 day",
            PairsAnalysis.coint_statistic.isnot(None)
        )
    ).scalar()
    
    with_adf_p = session.execute(
        select(func.count(PairsAnalysis.id))
        .where(
            PairsAnalysis.timeframe == "1 day",
            PairsAnalysis.adf_pvalue.isnot(None)
        )
    ).scalar()
    
    with_coint_p = session.execute(
        select(func.count(PairsAnalysis.id))
        .where(
            PairsAnalysis.timeframe == "1 day",
            PairsAnalysis.coint_pvalue.isnot(None)
        )
    ).scalar()
    
    print(f"Verification for timeframe '1 day':")
    print(f"  Total pairs: {total}")
    if total > 0:
        print(f"  Pairs with ADF statistic: {with_adf_stat} ({with_adf_stat/total*100:.1f}%)")
        print(f"  Pairs with ADF p-value: {with_adf_p} ({with_adf_p/total*100:.1f}%)")
        print(f"  Pairs with cointegration statistic: {with_coint_stat} ({with_coint_stat/total*100:.1f}%)")
        print(f"  Pairs with cointegration p-value: {with_coint_p} ({with_coint_p/total*100:.1f}%)")
    else:
        print("  No pairs found for this timeframe")
    
    # Show a sample pair
    if total > 0:
        sample = session.execute(
            select(PairsAnalysis)
            .where(PairsAnalysis.timeframe == "1 day")
            .limit(1)
        ).scalar_one()
        
        print(f"\nSample pair: {sample.symbol_a}/{sample.symbol_b}")
        print(f"  ADF statistic: {float(sample.adf_statistic) if sample.adf_statistic else 0:.6f}")
        print(f"  ADF p-value: {float(sample.adf_pvalue) if sample.adf_pvalue else 0:.6f}")
        print(f"  Cointegration statistic: {float(sample.coint_statistic) if sample.coint_statistic else 0:.6f}")
        print(f"  Cointegration p-value: {float(sample.coint_pvalue) if sample.coint_pvalue else 0:.6f}")

