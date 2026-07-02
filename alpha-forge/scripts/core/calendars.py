"""交易日历:从仅有 OHLC 值(丢失真实日期)的价格序列重建真实交易日。

当数据落盘时只存了价格、用 bdate_range 顶替日期会与真实交易日漂移(bdate_range 把节假日
也算成交易日)。本模块按 NYSE 口径重建,用于给买卖点标注真实日期(html_report 的 tradesChart
读取 trades['dates'] 后会在 K 线图标记上画 MM-DD)。
"""
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, nearest_workday,
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, USLaborDay, USThanksgivingDay)


class NYSECalendar(AbstractHolidayCalendar):
    """NYSE 假日:含 Good Friday;Columbus/Veterans Day 照常交易(故不在表内)。"""
    rules = [
        Holiday("NewYear", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday),
        Holiday("Independence", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_NYSE_CBD = pd.offsets.CustomBusinessDay(calendar=NYSECalendar())


def nyse_dates(end, periods):
    """返回截至 end(含)的最后 `periods` 个 NYSE 交易日(DatetimeIndex)。

    用法:已有长度为 N 的价格序列但日期不准时,
        df.index = nyse_dates("2026-06-25", len(df))
    再令 trades['dates'] = [d.strftime('%Y-%m-%d') for d in df.index],
    tradesChart 即会在买卖标记上标 MM-DD。
    """
    return pd.date_range(end=pd.Timestamp(end), periods=int(periods), freq=_NYSE_CBD)


def business_dates(end, periods, holidays=()):
    """截至 end 的最后 `periods` 个工作日,排除 `holidays`(可迭代的 'YYYY-MM-DD')。
    通用底座:任何市场都可传入自己的假日集合。"""
    hol = set(holidays)
    days = pd.bdate_range(end=pd.Timestamp(end), periods=int(periods) + len(hol) + 12)
    days = [d for d in days if d.strftime("%Y-%m-%d") not in hol]
    return pd.DatetimeIndex(days[-int(periods):])


# 港交所(SEHK)休市日 —— 已核对 2025 与 2026(含农历/复活节/佛诞等);如跨年请按 HKEX 日历补充。
SEHK_HOLIDAYS = {
    "2025-01-01", "2025-01-29", "2025-01-30", "2025-01-31", "2025-04-04", "2025-04-18",
    "2025-04-21", "2025-05-01", "2025-05-05", "2025-07-01", "2025-10-01", "2025-10-07",
    "2025-10-29", "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-02-17", "2026-02-18", "2026-02-19", "2026-04-03", "2026-04-06",
    "2026-04-07", "2026-05-01", "2026-05-25", "2026-06-19",
}


def sehk_dates(end, periods):
    """截至 end 的最后 `periods` 个港交所交易日(假日表覆盖 2025–2026)。"""
    return business_dates(end, periods, SEHK_HOLIDAYS)
