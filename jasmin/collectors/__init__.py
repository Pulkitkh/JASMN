from jasmin.collectors.prices import PriceCollector
from jasmin.collectors.fundamentals import FundamentalsCollector
from jasmin.collectors.macro import MacroCollector
from jasmin.collectors.institutional import InstitutionalCollector
from jasmin.collectors.news import NewsCollector

ALL_COLLECTORS = [
    PriceCollector,
    FundamentalsCollector,
    MacroCollector,
    InstitutionalCollector,
    NewsCollector,
]
