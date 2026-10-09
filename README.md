# A 股多策略复盘台

一个本地运行的 A 股复盘 Web 工具，目前包含两种策略：

- 主线复盘策略：市场温度、板块强度、个股候选、风控提示。
- 涨停板策略：支持 `1-1`、`1-2`、`2-` 这类连板范围筛选，默认用实换手率过滤。
- 1进2实时判断：内嵌在涨停板策略页，用前一交易日首板候选叠加早盘实时行情，输出动作、买点、理由和失效条件。
- 欧奈尔模型Q0研究观察池：Tushare主源、AkShare兜底，逐条显示原书/推导/A股适配/待确认来源；当前不提供组合回测。

## 运行

Windows 登录后，控制后台会静默运行，不会自动打开浏览器。

需要操作时，直接双击：

```text
D:\quant\quant_control.html
```

这个本地 HTML 页面可以启动、停止复盘服务，或打开复盘网页。

控制后台地址：

```text
http://127.0.0.1:8764
```

也可以直接运行：

```powershell
python app.py
```

浏览器访问：

```text
http://127.0.0.1:8765
```

欧奈尔模型页面：

```text
http://127.0.0.1:8765/oneil-model
```

## 目录结构

```text
quant_review/
  config.py                  # 页面参数和策略阈值配置
  engine.py                  # Tushare 数据读取、缓存、通用数据接口
  server.py                  # HTTP 路由和 API
  templates/
    index.html               # 主线复盘页面
    first_limit.html         # 涨停板策略页面
  strategies/
    mainline_review.py       # 主线复盘策略
    first_limit.py           # 涨停板策略
    one_to_two_live.py       # 1进2早盘实时判断

app.py                       # 推荐启动入口
requirements.txt             # 运行依赖
```

## 新增策略

新增策略时，优先在 `quant_review/strategies/` 里加一个独立文件，
再在 `quant_review/server.py` 里增加对应 API 和页面路由。
