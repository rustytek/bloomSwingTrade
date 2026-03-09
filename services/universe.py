"""
S&P 500 constituents (as of early 2025) + major ETFs.
This defines the screener universe.  Update periodically as the index changes.
"""

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
    # International
    "EFA","EEM","FXI","EWJ","EWZ","MCHI","INDA","EWU","EWG","EWC",
    # Dividend / factor
    "DVY","VYM","SDY","NOBL","HDV","DGRO",
    "IWF","IWD","VUG","VTV","MTUM","VLUE",
    # Leveraged (popular)
    "TQQQ","SOXL","UPRO","SPXL","TECL",
    # Volatility / alternatives
    "UVXY","GDX","GDXJ",
]

# Deduplicated combined universe
UNIVERSE: list[str] = sorted(set(SP500 + ETFS))
