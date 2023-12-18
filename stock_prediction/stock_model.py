from typing import Optional, Any, Dict, TypeVar, Callable, Type, cast


T = TypeVar("T")


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def to_float(x: Any) -> float:
    assert isinstance(x, float)
    return x


def from_dict(f: Callable[[Any], T], x: Any) -> Dict[str, T]:
    assert isinstance(x, dict)
    return { k: f(v) for (k, v) in x.items() }


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


class Item:
    date: str
    dateutc: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjclose: Optional[float]

    def __init__(self, date: str, dateutc: int, open: float, high: float, low: float, close: float, volume: int, adjclose: Optional[float]) -> None:
        self.date = date
        self.dateutc = dateutc
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.adjclose = adjclose

    @staticmethod
    def from_dict(obj: Any) -> 'Item':
        assert isinstance(obj, dict)
        date = from_str(obj.get("date"))
        dateutc = from_int(obj.get("date_utc"))
        open = from_float(obj.get("open"))
        high = from_float(obj.get("high"))
        low = from_float(obj.get("low"))
        close = from_float(obj.get("close"))
        volume = from_int(obj.get("volume"))
        adjclose = from_union([from_float, from_none], obj.get("adjclose"))
        return Item(date, dateutc, open, high, low, close, volume, adjclose)

    def to_dict(self) -> dict:
        result: dict = {}
        result["date"] = from_str(self.date)
        result["date_utc"] = from_int(self.dateutc)
        result["open"] = to_float(self.open)
        result["high"] = to_float(self.high)
        result["low"] = to_float(self.low)
        result["close"] = to_float(self.close)
        result["volume"] = from_int(self.volume)
        if self.adjclose is not None:
            result["adjclose"] = from_union([to_float, from_none], self.adjclose)
        return result


class Meta:
    currency: str
    symbol: str
    exchangeName: str
    instrumentType: str
    firstTradeDate: int
    regularMarketTime: int
    gmtoffset: int
    timezone: str
    exchangeTimezoneName: str
    regularMarketPrice: float
    chartPreviousClose: float
    priceHint: int
    dataGranularity: str
    range: str

    def __init__(self, currency: str, symbol: str, exchangeName: str, instrumentType: str, firstTradeDate: int, regularMarketTime: int, gmtoffset: int, timezone: str, exchangeTimezoneName: str, regularMarketPrice: float, chartPreviousClose: float, priceHint: int, dataGranularity: str, range: str) -> None:
        self.currency = currency
        self.symbol = symbol
        self.exchangeName = exchangeName
        self.instrumentType = instrumentType
        self.firstTradeDate = firstTradeDate
        self.regularMarketTime = regularMarketTime
        self.gmtoffset = gmtoffset
        self.timezone = timezone
        self.exchangeTimezoneName = exchangeTimezoneName
        self.regularMarketPrice = regularMarketPrice
        self.chartPreviousClose = chartPreviousClose
        self.priceHint = priceHint
        self.dataGranularity = dataGranularity
        self.range = range

    @staticmethod
    def from_dict(obj: Any) -> 'Meta':
        assert isinstance(obj, dict)
        currency = from_str(obj.get("currency"))
        symbol = from_str(obj.get("symbol"))
        exchangeName = from_str(obj.get("exchangeName"))
        instrumentType = from_str(obj.get("instrumentType"))
        firstTradeDate = from_int(obj.get("firstTradeDate"))
        regularMarketTime = from_int(obj.get("regularMarketTime"))
        gmtoffset = from_int(obj.get("gmtoffset"))
        timezone = from_str(obj.get("timezone"))
        exchangeTimezoneName = from_str(obj.get("exchangeTimezoneName"))
        regularMarketPrice = from_float(obj.get("regularMarketPrice"))
        chartPreviousClose = from_float(obj.get("chartPreviousClose"))
        priceHint = from_int(obj.get("priceHint"))
        dataGranularity = from_str(obj.get("dataGranularity"))
        range = from_str(obj.get("range"))
        return Meta(currency, symbol, exchangeName, instrumentType, firstTradeDate, regularMarketTime, gmtoffset, timezone, exchangeTimezoneName, regularMarketPrice, chartPreviousClose, priceHint, dataGranularity, range)

    def to_dict(self) -> dict:
        result: dict = {}
        result["currency"] = from_str(self.currency)
        result["symbol"] = from_str(self.symbol)
        result["exchangeName"] = from_str(self.exchangeName)
        result["instrumentType"] = from_str(self.instrumentType)
        result["firstTradeDate"] = from_int(self.firstTradeDate)
        result["regularMarketTime"] = from_int(self.regularMarketTime)
        result["gmtoffset"] = from_int(self.gmtoffset)
        result["timezone"] = from_str(self.timezone)
        result["exchangeTimezoneName"] = from_str(self.exchangeTimezoneName)
        result["regularMarketPrice"] = to_float(self.regularMarketPrice)
        result["chartPreviousClose"] = to_float(self.chartPreviousClose)
        result["priceHint"] = from_int(self.priceHint)
        result["dataGranularity"] = from_str(self.dataGranularity)
        result["range"] = from_str(self.range)
        return result


class Stock:
    meta: Meta
    items: Dict[str, Item]
    error: None

    def __init__(self, meta: Meta, items: Dict[str, Item], error: None) -> None:
        self.meta = meta
        self.items = items
        self.error = error

    @staticmethod
    def from_dict(obj: Any) -> 'Stock':
        assert isinstance(obj, dict)
        meta = Meta.from_dict(obj.get("meta"))
        items = from_dict(Item.from_dict, obj.get("body"))
        error = from_none(obj.get("error"))
        return Stock(meta, items, error)

    def to_dict(self) -> dict:
        result: dict = {}
        result["meta"] = to_class(Meta, self.meta)
        result["items"] = from_dict(lambda x: to_class(Item, x), self.items)
        result["error"] = from_none(self.error)
        return result


def Stockfromdict(s: Any) -> Stock:
    return Stock.from_dict(s)


def Stocktodict(x: Stock) -> Any:
    return to_class(Stock, x)
