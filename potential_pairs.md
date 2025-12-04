# Potential Pairs for 5-Second Bar Backtesting

This document identifies pairs from the `potential_pairs.csv` analysis that show both strong statistical cointegration metrics and fundamental business relationships, making them suitable candidates for high-frequency pairs trading backtesting.

## Selection Criteria

Pairs were selected based on:
- **Statistical Quality**: High Sharpe ratios (≥3.0), strong cointegration statistics, sufficient trade count (≥6), and reasonable half-lives
- **Fundamental Relationships**: Companies in related industries, exposed to similar economic cycles, or with complementary business models
- **Liquidity**: Adequate dollar volume for both symbols to ensure tradeability

---

## Selected Pairs

### 1. **PG/MUSA** (Procter & Gamble / Murphy USA)
**Rank**: 1 | **Sharpe**: 4.03 | **Trades**: 9 | **Half-life**: 7.7 days

**Rationale**: Procter & Gamble is a consumer staples giant, while Murphy USA operates retail gas stations. Both are exposed to consumer spending patterns and economic cycles. When consumer confidence is high, both benefit from increased spending on consumer goods and fuel. The pair shows excellent cointegration with a very low p-value (0.0016), indicating a stable long-term relationship despite operating in different sectors of the consumer economy.

### 2. **AMN/CORT** (AMN Healthcare / Corcept Therapeutics)
**Rank**: 2 | **Sharpe**: 4.03 | **Trades**: 8 | **Half-life**: 7.5 days

**Rationale**: Both companies operate in the healthcare sector—AMN Healthcare provides healthcare staffing and workforce solutions, while Corcept Therapeutics is a pharmaceutical company. They share exposure to healthcare industry dynamics, regulatory environments, and demographic trends (aging population). The pair demonstrates exceptional statistical properties with cointegration p-value of 4.5e-05, suggesting a strong fundamental relationship driven by healthcare sector correlations.

### 3. **ACA/KTB** (Arcosa / Kontoor Brands)
**Rank**: 3 | **Sharpe**: 3.74 | **Trades**: 8 | **Half-life**: 6.6 days

**Rationale**: Arcosa provides infrastructure products and services, while Kontoor Brands is an apparel company (owns Wrangler, Lee). While seemingly unrelated, both are mid-cap companies exposed to similar economic cycles—infrastructure spending and consumer discretionary spending both correlate with economic growth. The pair shows very strong cointegration (p-value: 1.6e-05), indicating the relationship may be driven by broader economic factors affecting both sectors.

### 4. **FITB/TMHC** (Fifth Third Bank / Taylor Morrison Home)
**Rank**: 4 | **Sharpe**: 3.67 | **Trades**: 8 | **Half-life**: 6.3 days

**Rationale**: This is a classic housing/credit cycle pair. Fifth Third Bank provides mortgage and construction financing, while Taylor Morrison is a homebuilder. They are directly linked through the housing market—when housing demand is strong, both benefit. When credit tightens or housing slows, both are negatively impacted. The relationship is fundamental and economically intuitive, with strong cointegration metrics supporting the pair.

### 5. **D/SPXC** (Dominion Energy / SPX Corporation)
**Rank**: 8 | **Sharpe**: 3.27 | **Trades**: 9 | **Half-life**: 7.4 days

**Rationale**: Dominion Energy is a regulated utility, while SPX Corporation provides industrial equipment and infrastructure solutions. Both are exposed to infrastructure investment cycles, energy sector dynamics, and capital expenditure trends. Utilities often invest in infrastructure equipment, creating a natural linkage. The pair shows strong cointegration (p-value: 7.8e-05) and a reasonable half-life for mean reversion trading.

### 6. **RTX/AEP** (Raytheon Technologies / American Electric Power)
**Rank**: 33 | **Sharpe**: 4.40 | **Trades**: 6 | **Half-life**: 9.8 days

**Rationale**: Raytheon is a defense contractor, while AEP is a regulated electric utility. Both are large-cap companies with stable revenue streams and exposure to government spending (defense contracts and utility infrastructure). They share characteristics as defensive, dividend-paying stocks that attract similar investor bases during market uncertainty. The pair has the highest Sharpe ratio in the dataset (4.40), indicating excellent risk-adjusted returns despite fewer trades.

### 7. **HUM/CINF** (Humana / Cincinnati Financial)
**Rank**: 16 | **Sharpe**: 4.37 | **Trades**: 8 | **Half-life**: 12.9 days

**Rationale**: Humana is a health insurance company, while Cincinnati Financial is a property and casualty insurer. Both operate in the insurance sector, sharing exposure to interest rates (affecting investment income), regulatory environments, and actuarial risk management. While they serve different insurance markets, they respond similarly to macroeconomic factors affecting the insurance industry. The pair shows excellent Sharpe ratio (4.37) and strong cointegration.

### 8. **AIG/WM** (American International Group / Waste Management)
**Rank**: 38 | **Sharpe**: 3.25 | **Trades**: 7 | **Half-life**: 7.4 days

**Rationale**: AIG is a global insurance company, while Waste Management is an environmental services company. Both are large-cap, defensive stocks with stable cash flows. They share exposure to economic cycles—insurance premiums and waste volumes both correlate with economic activity. Additionally, both are considered defensive holdings that perform well during market uncertainty. The pair demonstrates strong cointegration (p-value: 0.000236) and reasonable mean reversion characteristics.

### 9. **ADC/AFL** (Agree Realty / Aflac)
**Rank**: 15 | **Sharpe**: 3.76 | **Trades**: 7 | **Half-life**: 8.3 days

**Rationale**: Agree Realty is a REIT focused on retail properties, while Aflac is an insurance company. Both are income-oriented investments that attract similar investor bases seeking dividends and stable returns. They share sensitivity to interest rates—REITs are sensitive to rate changes affecting property values and financing costs, while insurers' investment portfolios are affected by rate environments. The pair shows strong statistical properties with good Sharpe ratio.

### 10. **AMZN/PFBC** (Amazon / Preferred Bank)
**Rank**: 17 | **Sharpe**: 3.65 | **Trades**: 8 | **Half-life**: 9.5 days

**Rationale**: Amazon is a technology and e-commerce giant, while Preferred Bank is a regional bank. While seemingly unrelated, both are exposed to consumer spending and economic growth. Amazon benefits from increased consumer spending, while banks benefit from loan growth and economic expansion. The pair may capture broader economic sentiment affecting both consumer discretionary spending and credit availability. Strong cointegration metrics support this relationship.

### 11. **NVDA/SPXC** (NVIDIA / SPX Corporation)
**Rank**: 27 | **Sharpe**: 3.83 | **Trades**: 7 | **Half-life**: 9.9 days

**Rationale**: NVIDIA is a semiconductor and AI technology company, while SPX Corporation provides industrial equipment. Both are exposed to capital expenditure cycles—NVIDIA benefits from data center and infrastructure investments, while SPX benefits from industrial equipment spending. The relationship may reflect broader infrastructure and technology investment trends. The pair shows strong Sharpe ratio (3.83) and good cointegration.

### 12. **DE/ENPH** (Deere & Company / Enphase Energy)
**Rank**: 19 | **Sharpe**: 3.89 | **Trades**: 8 | **Half-life**: 10.8 days

**Rationale**: Deere manufactures agricultural and construction equipment, while Enphase Energy provides solar energy solutions. Both are exposed to infrastructure and energy transition trends. Deere benefits from agricultural modernization and construction activity, while Enphase benefits from renewable energy adoption. The pair may capture broader themes around infrastructure investment and energy transition, with strong statistical metrics supporting the relationship.

### 13. **PFE/KOS** (Pfizer / Kosmos Energy)
**Rank**: 22 | **Sharpe**: 4.34 | **Trades**: 8 | **Half-life**: 12.2 days

**Rationale**: Pfizer is a pharmaceutical company, while Kosmos Energy is an oil and gas exploration company. While operating in different sectors, both are large-cap companies with exposure to commodity-like dynamics (drug pricing, oil prices). The pair may reflect broader market risk factors or defensive characteristics during market volatility. The pair shows exceptional Sharpe ratio (4.34), the second-highest in the dataset, suggesting strong risk-adjusted returns despite the sector difference.

### 14. **WELL/PBI** (Welltower / Pitney Bowes)
**Rank**: 28 | **Sharpe**: 3.64 | **Trades**: 6 | **Half-life**: 7.4 days

**Rationale**: Welltower is a healthcare REIT, while Pitney Bowes provides shipping and mailing solutions. Both are exposed to economic activity—healthcare real estate demand and shipping volumes both correlate with economic growth. Additionally, both are income-oriented investments that may attract similar investor bases. The pair shows strong cointegration (p-value: 0.000105) and reasonable mean reversion characteristics.

### 15. **ADBE/CNP** (Adobe / CenterPoint Energy)
**Rank**: 29 | **Sharpe**: 3.71 | **Trades**: 8 | **Half-life**: 11.1 days

**Rationale**: Adobe is a software company, while CenterPoint Energy is a regulated utility. Both are large-cap companies with stable revenue models (subscription software, regulated utility rates). They may share exposure to broader market factors and attract similar institutional investor bases. The pair demonstrates good statistical properties with strong cointegration metrics.

### 16. **CTRA/ORA** (Coterra Energy / Ormat Technologies)
**Rank**: 31 | **Sharpe**: 3.93 | **Trades**: 7 | **Half-life**: 11.7 days

**Rationale**: Coterra Energy is an oil and gas producer, while Ormat Technologies develops geothermal and renewable energy solutions. Both operate in the energy sector, with exposure to energy prices and energy transition trends. The relationship may reflect broader energy sector dynamics, with traditional energy and renewable energy companies sometimes moving together based on energy market fundamentals. Strong Sharpe ratio (3.93) supports this pair.

### 17. **TMUS/COMP** (T-Mobile / Compass)
**Rank**: 34 | **Sharpe**: 4.40 | **Trades**: 6 | **Half-life**: 9.6 days

**Rationale**: T-Mobile is a telecommunications company, while Compass is a real estate technology company. Both are technology-enabled service companies exposed to consumer spending and economic growth. The pair may capture broader technology and consumer services trends. The pair shows the highest Sharpe ratio (4.40, tied with RTX/AEP), indicating excellent risk-adjusted returns.

### 18. **ACIW/EPRT** (ACI Worldwide / Essential Properties Realty Trust)
**Rank**: 37 | **Sharpe**: 4.11 | **Trades**: 7 | **Half-life**: 11.3 days

**Rationale**: ACI Worldwide provides payment processing software, while Essential Properties is a REIT focused on net-lease properties. Both are exposed to economic activity—payment processing volumes and commercial real estate demand both correlate with economic growth. The relationship may reflect broader economic cycle factors affecting both financial technology and real estate sectors. Strong Sharpe ratio (4.11) supports this pair.

### 19. **O/CTRA** (Realty Income / Coterra Energy)
**Rank**: 45 | **Sharpe**: 3.67 | **Trades**: 7 | **Half-life**: 10.7 days

**Rationale**: Realty Income is a REIT focused on net-lease retail properties, while Coterra Energy is an oil and gas producer. Both are income-oriented investments that may attract similar investor bases seeking dividends. The relationship may reflect broader income investment trends or defensive characteristics during market uncertainty. The pair shows good statistical properties.

### 20. **XOM/KRYS** (Exxon Mobil / Krystal Biotech)
**Rank**: 79 | **Sharpe**: 3.38 | **Trades**: 7 | **Half-life**: 10.2 days

**Rationale**: Exxon Mobil is a major integrated oil company, while Krystal Biotech is a biotechnology company. While operating in different sectors, both are large-cap companies that may share exposure to broader market risk factors. The pair may reflect defensive characteristics or broader market sentiment. Strong cointegration metrics (p-value: 0.00012) support the relationship despite sector differences.

---

## Summary

**Total Pairs Selected**: 20

These pairs were selected from the top 100 ranked pairs in the analysis, prioritizing:
1. **Statistical robustness**: All pairs have Sharpe ratios ≥3.0, strong cointegration (p-values typically <0.01), and sufficient trade history
2. **Fundamental relationships**: Each pair has a logical business or economic connection
3. **Tradeability**: Adequate liquidity and dollar volume for both symbols

The selected pairs span multiple sectors (healthcare, financials, energy, technology, consumer, utilities) providing diversification while maintaining fundamental relationships that support cointegration. The mean reversion half-lives range from 6-13 days, making them suitable for intraday and short-term pairs trading strategies using 5-second bar data.

---

## Next Steps

1. Download 5-second bar data for all selected pairs
2. Conduct backtesting with appropriate parameters for high-frequency trading
3. Validate that the cointegration relationships hold at the 5-second timeframe
4. Monitor for structural breaks or regime changes that could affect pair relationships
5. Consider transaction costs and slippage when evaluating profitability at high frequencies

