"""
S&P 500 constituents (as of early 2025) + major ETFs.
This defines the screener universe.  Update periodically as the index changes.

SURVIVORSHIP BIAS — READ THIS BEFORE TRUSTING ANY BACKTEST NUMBER
-----------------------------------------------------------------
This list is TODAY's index membership, not point-in-time membership. It is a
perfectly good *screener* universe and a systematically optimistic *backtest*
universe, because every name in it is one that:
  - still exists (it did not go bankrupt, get acquired, or delist), and
  - is still in the index (it was not dropped for poor performance).

Any historical test run over this list therefore only ever trades the winners,
chosen with hindsight the strategy did not have at the time. CAGR, win rate and
expectancy all come out too high. The distortion is worst for long lookbacks and
for strategies that buy weakness (a falling name that later recovered is in the
list; one that went to zero never appears).

We do NOT attempt to reconstruct point-in-time index membership: yfinance does
not serve historical constituent lists, and there is no free source wired into
this app. The honest response is disclosure, not a silent fix — so every
consumer that reports historical performance renders UNIVERSE_CAVEAT.

Consumers: services/backtest.py (caveats block), services/edge_matrix.py,
services/scorecard.py, api/scorecard.py.
"""

# When this constituent list was last curated. Rendered alongside UNIVERSE_CAVEAT
# so a stale list is visible rather than implied.
UNIVERSE_AS_OF = "2025-01"

UNIVERSE_CAVEAT = (
    "The ticker universe is TODAY's S&P 500 / ETF constituent list "
    f"(services/universe.py, curated {UNIVERSE_AS_OF}), not point-in-time index "
    "membership. Companies that went bankrupt, were acquired, or were dropped "
    "from the index never appear, so any historical result only ever trades names "
    "that survived AND stayed in the index. Returns, win rates and expectancy are "
    "all biased upward — treat them as an optimistic ceiling, not a forecast. "
    "Point-in-time membership is not available through yfinance, so this bias is "
    "disclosed rather than corrected."
)

SP500 = [
    # ── Technology ──────────────────────────────────────────────────────────
    "AAPL","MSFT","NVDA","AVGO","ORCL","CSCO","ACN","AMD","ADBE","INTC",
    "TXN","QCOM","INTU","IBM","NOW","AMAT","LRCX","KLAC","ADI","SNPS",
    "CDNS","PANW","CRM","FTNT","WDAY","MU","HPE","CDW","TER",
    "SMCI","NXPI","MCHP","ON","SWKS","MPWR","PAYC","CTSH","IT","GRMN",
    "PTC","PLTR","DELL","HPQ","TEL","TYL","JKHY","EPAM","KEYS","AKAM",
    "APP","GEV",                                   # added 2024-2025
    # ── Communication Services ───────────────────────────────────────────────
    "GOOGL","GOOG","META","NFLX","DIS","CMCSA","VZ","T","TMUS","CHTR",
    "WBD","OMC","FOXA","FOX","NWSA","NWS","TTWO","EA",
    "MTCH","LYV",
    # removed: PARA (acquired by Skydance 2024), IPG (acquired by Omnicom 2025)
    # ── Consumer Discretionary ──────────────────────────────────────────────
    "AMZN","TSLA","HD","MCD","NKE","SBUX","TJX","BKNG","LOW","ABNB",
    "MAR","HLT","CMG","LULU","F","GM","ORLY","AZO","ROST","YUM",
    "DHI","LEN","PHM","NVR","TOL","BBY","ULTA","DRI","TGT","TSCO",
    "EXPE","RCL","CCL","NCLH","MGM","LVS","CZR","WYNN","TPR","RL",
    "PVH","HAS","MAT","WHR","BWA","APTV","MHK","NWL","DKNG","DECK",
    # ── Consumer Staples ────────────────────────────────────────────────────
    "PG","KO","PEP","COST","WMT","PM","MO","MDLZ","CL","GIS",
    "KHC","HSY","CHD","SJM","CAG","MKC","CPB","HRL","TSN",
    "KR","KVUE","KDP","MNST","TAP","STZ","EL","CLX","SYY","SFM",
    # removed: K (Kellanova acquired by Mars 2024), WBA (went private 2024)
    # ── Energy ──────────────────────────────────────────────────────────────
    "XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","DVN",
    "FANG","APA","HAL","BKR","KMI","WMB","OKE","LNG",
    "TRGP","EQT","RRC","AR","VST",
    # removed: HES (acquired by CVX 2024), MRO (acquired by COP 2024)
    # ── Financials ──────────────────────────────────────────────────────────
    "BRK-B","JPM","V","MA","BAC","WFC","GS","MS","AXP","BLK",
    "SCHW","C","USB","PNC","TFC","COF","AIG","MET","PRU",
    "AFL","ALL","PGR","TRV","CB","MMC","AON","SPGI","MCO","ICE",
    "CME","NDAQ","BK","STT","TROW","IVZ","BEN","AMP","RJF",
    "HBAN","RF","KEY","FITB","CFG","MTB","ZION","WRB","ACGL","AIZ",
    "FIS","GPN","CPAY","SYF","ALLY",
    # removed: DFS (acquired by COF 2024), FI (delisted/invalid)
    # ── Health Care ─────────────────────────────────────────────────────────
    "LLY","UNH","JNJ","MRK","ABBV","TMO","ABT","DHR","BMY","AMGN",
    "PFE","GILD","REGN","VRTX","CI","HUM","MCK","CVS","ELV","ZBH",
    "BDX","BSX","SYK","MDT","ISRG","EW","RMD","DXCM","IQV","MTD",
    "A","WAT","HOLX","TECH","BIO","IDXX","PODD","RVTY","DGX","LH",
    "VTRS","MRNA","BIIB","ALNY","INCY","ILMN","CRL",
    "HCA","UHS","MOH","CNC","DVA","STE","HSIC",
    # removed: CTLT (acquired by Novo Holdings 2024), ANSS (acquired by SNPS 2024)
    # ── Industrials ─────────────────────────────────────────────────────────
    "CAT","DE","HON","RTX","LMT","GE","NOC","GD","BA","UPS",
    "FDX","CSX","NSC","UNP","WM","RSG","EMR","ETN","PH","ITW",
    "MMM","ROK","AME","CTAS","PAYX","FAST","SWK","IR","XYL","IEX",
    "VRSK","CPRT","ODFL","SAIA","GNRC","JCI","TT","CARR","OTIS",
    "HWM","TDG","TXT","LHX","LDOS","HUBB","NDSN","ROP","TRMB",
    "JBHT","EXPD","WAB","AXON","GWW","PCAR","CMI","URI",
    "DAL","UAL","LUV","AAL","ALK","CHRW","J",
    # ── Materials ───────────────────────────────────────────────────────────
    "LIN","APD","SHW","ECL","DD","DOW","NEM","FCX","NUE","STLD",
    "ALB","CF","MOS","FMC","IFF","PPG","RPM","VMC","MLM","SW",
    "PKG","IP","SEE","AVY","SON","BALL","CCK","OLN","EMN","CE",
    # removed: WRK (merged into SW / Smurfit WestRock 2024)
    # ── Real Estate ─────────────────────────────────────────────────────────
    "AMT","PLD","CCI","EQIX","PSA","SPG","O","WELL","DLR","AVB",
    "EQR","VTR","VICI","CBRE","ARE","BXP","KIM","REG","FRT","WPC",
    "EXR","INVH","ESS","MAA","UDR","CPT","HST","DOC","SBAC","AMH",
    # removed: SBA (duplicate of SBAC — SBA Communications trades as SBAC)
    # ── Utilities ───────────────────────────────────────────────────────────
    "NEE","DUK","SO","D","AEP","EXC","XEL","WEC","ES","AWK",
    "ED","FE","ETR","CMS","NI","PPL","LNT","EVRG","PNW","ATO",
    "SRE","PCG","AEE","CNP","NRG","CEG",
]

ETFS = [
    # Broad market
    "SPY","QQQ","IWM","DIA","VTI","VOO","VTWO","VEA","VWO","VT",
    # SPDR sectors
    "XLK","XLF","XLV","XLE","XLI","XLY","XLP","XLU","XLRE","XLB","XLC",
    # Vanguard sectors
    "VGT","VFH","VHT","VDE","VIS","VCR","VDC","VPU",
    # Thematic / sector
    "IBB","XBI","SOXX","SMH","HACK","ICLN","WCLD",
    "ARKK","ARKG","ARKW","ARKF",
    # Fixed income
    "TLT","IEF","SHY","HYG","LQD","BND","AGG","BNDX","VCSH","VCIT",
    # Commodities
    "GLD","SLV","USO","UNG","DBA","PDBC","IAU",
    "GSG","COMT","DBC","GCC","USCI",
    # International
    "EFA","EEM","FXI","EWJ","EWZ","MCHI","INDA","EWU","EWG","EWC",
    "VXUS","FDT","IEMG","AVDV","SCHY","EFV","VGK","VPL",
    # Dividend / factor
    "DVY","VYM","SDY","NOBL","HDV","DGRO",
    "IWF","IWD","VUG","VTV","MTUM","VLUE",
    # Style / size
    "SPYG","SPYV","MDYG","MDYV","SLYG","SLYV","IWC",
    # Leveraged (popular)
    "TQQQ","SOXL","UPRO","SPXL","TECL",
    # Volatility / alternatives
    "UVXY","GDX","GDXJ",
]

# Deduplicated combined universe
UNIVERSE: list[str] = sorted(set(SP500 + ETFS))
