# 导入tushare
import tushare as ts


# 初始化pro接口
pro = ts.pro_api('13bf507ca254cc8a32a07dee082e4e327b4fb20b1f0186bb05692e8a')

# 拉取数据
df = pro.daily(**{
    "ts_code": "600588.SH",
    "trade_date": "",
    "start_date": 20240606,
    "end_date": 20240612,
    "offset": "",
    "limit": ""
}, fields=[
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount"
])
print(df)

