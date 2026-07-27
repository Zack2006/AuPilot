"""Shared Databento daily chart powered by vendored Lightweight Charts 5.2.0."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st

from frontend.i18n import tr


UP_COLOR = "#C84B4B"
DOWN_COLOR = "#238B57"
GRID_COLOR = "#D9DEE4"
HISTORICAL_SIGNAL_COLORS = {
    "TOP": "#D25760",
    "TOP_L1": "#E79AA2",
    "TOP_L2": "#D25760",
    "TOP_L3": "#A52E38",
    "BOTTOM": "#3EA66D",
    "BOTTOM_L1": "#83D4A6",
    "BOTTOM_L2": "#3EA66D",
    "BOTTOM_L3": "#17683F",
}
SMA_COLORS = {
    "SMA5": "#2F6B9A",
    "SMA10": "#D1872C",
    "SMA20": "#6E5AA8",
    "SMA60": "#278A8A",
    "SMA120": "#8B5D3B",
    "SMA250": "#555E68",
}
_LIBRARY_SHA256 = "C0992580867C4912CC9385B3C2728315BCC1A76C7F1087DCA908430FCCDF31D7"
_LIBRARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "vendor"
    / "lightweight_charts"
    / "lightweight-charts.standalone.production.js"
)


def market_frame(payload: dict) -> pd.DataFrame:
    """Validate the frontend boundary without recalculating backend indicators."""
    frame = pd.DataFrame(payload.get("items", []))
    required = {
        "ts_event_utc", "open", "high", "low", "close", "volume", "change_amount_1d", "return_1d",
        "sma_5", "sma_10", "sma_20", "sma_60", "sma_120", "sma_250",
        "boll_mid", "boll_upper", "boll_lower", "macd_dif", "macd_dea", "macd_hist",
        "rsi_14", "kdj_k", "kdj_d", "kdj_j", "atr_14", "volatility_20",
        "drawdown_from_high",
    }
    missing = required - set(frame.columns)
    if frame.empty or missing:
        raise ValueError(f"Incomplete Databento technical-history payload: {sorted(missing)}")
    frame["ts_event_utc"] = pd.to_datetime(frame["ts_event_utc"], utc=True, errors="raise")
    numeric = sorted(required - {"ts_event_utc"})
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_values("ts_event_utc").drop_duplicates("ts_event_utc", keep=False)
    if frame.empty or not frame["ts_event_utc"].is_monotonic_increasing:
        raise ValueError("Databento chart timestamps are not unique and increasing")
    return frame.reset_index(drop=True)


@lru_cache(maxsize=1)
def _library_source() -> str:
    raw = _LIBRARY_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest().upper() != _LIBRARY_SHA256:
        raise ValueError("Vendored Lightweight Charts asset failed SHA-256 verification")
    return raw.decode("utf-8")


def _date(value: pd.Timestamp) -> str:
    return value.date().isoformat()


def _line(frame: pd.DataFrame, column: str) -> list[dict]:
    selected = frame.loc[frame[column].notna(), ["ts_event_utc", column]]
    return [
        {"time": _date(row.ts_event_utc), "value": float(getattr(row, column))}
        for row in selected.itertuples(index=False)
    ]


def _markers(frame: pd.DataFrame, issuances: list[dict] | None) -> list[dict]:
    if not issuances:
        return []
    market_dates = {_date(value) for value in frame["ts_event_utc"]}
    markers: list[dict] = []
    action_styles = {
        "HOLD": ("inBar", "circle", "#2F6B9A", "HOLD"),
        "REDUCE_GOLD_WEIGHT": (
            "aboveBar",
            "arrowDown",
            UP_COLOR,
            "REDUCE",
        ),
        "INCREASE_GOLD_WEIGHT": (
            "belowBar",
            "arrowUp",
            DOWN_COLOR,
            "INCREASE",
        ),
    }
    for record in issuances:
        output = record.get("output") or {}
        if record.get("status") != "SUCCESS" or not output:
            continue
        source_bucket = record.get("source_bucket")
        action = (output.get("action") or {}).get("action")
        if source_bucket in market_dates and action in action_styles:
            position, shape, color, text = action_styles[action]
            markers.append(
                {
                    "time": source_bucket,
                    "position": position,
                    "shape": shape,
                    "color": color,
                    "text": text,
                }
            )
        for slot in output.get("slots", []):
            target = slot.get("target_bucket")
            display_class = slot.get("display_class")
            if target not in market_dates:
                continue
            if isinstance(display_class, str) and display_class.startswith("TOP_"):
                markers.append(
                    {
                        "time": target,
                        "position": "aboveBar",
                        "shape": "arrowDown",
                        "color": UP_COLOR,
                        "text": display_class,
                    }
                )
            elif (
                isinstance(display_class, str)
                and display_class.startswith("BOTTOM_")
            ):
                markers.append(
                    {
                        "time": target,
                        "position": "belowBar",
                        "shape": "arrowUp",
                        "color": DOWN_COLOR,
                        "text": display_class,
                    }
                )
    return sorted(markers, key=lambda item: item["time"])


def _historical_markers(
    frame: pd.DataFrame,
    signals: list[dict] | None,
) -> list[dict]:
    if not signals:
        return []
    market_dates = {_date(value) for value in frame["ts_event_utc"]}
    markers: list[dict] = []
    for signal in signals:
        target = str(signal.get("signal_date") or "")
        label = str(signal.get("signal_label") or "")
        if target not in market_dates or label not in HISTORICAL_SIGNAL_COLORS:
            continue
        top = label.startswith("TOP_")
        action = str(signal.get("action") or "")
        executed = action in {
            "REDUCE_GOLD_WEIGHT",
            "REENTER_GOLD_WEIGHT",
        }
        markers.append(
            {
                "time": target,
                "position": "aboveBar" if top else "belowBar",
                "shape": "arrowDown" if top else "arrowUp",
                "color": (
                    HISTORICAL_SIGNAL_COLORS[label]
                    if executed
                    else f"{HISTORICAL_SIGNAL_COLORS[label]}55"
                ),
                "text": (
                    (
                        f"{'T' if top else 'B'}{label[-1]}"
                        if "_L" in label
                        else ("T" if top else "B")
                    )
                    if executed
                    else ""
                ),
                "size": 0.8 if executed else 0.4,
            }
        )
    return sorted(markers, key=lambda item: item["time"])


def _comparison_curve(rows: list[dict] | None) -> dict | None:
    if not rows:
        return None
    model: list[dict] = []
    buy_hold: list[dict] = []
    for row in rows:
        trade_date = str(row["trade_date"])
        model.append(
            {"time": trade_date, "value": float(row["model_return"])}
        )
        buy_hold.append(
            {"time": trade_date, "value": float(row["buy_hold_return"])}
        )
    return {"model": model, "buyHold": buy_hold}


def _future_forecast(forecast: dict | None) -> dict | None:
    if (
        not forecast
        or forecast.get("stale")
        or forecast.get("slot_count") != 21
        or len(forecast.get("slots", [])) != 21
    ):
        return None
    data: list[dict] = []
    metadata: dict[str, dict] = {}
    top_closes: list[dict] = []
    bottom_closes: list[dict] = []
    for slot in forecast["slots"]:
        target = str(slot["target_bucket"])
        signal = slot.get("display_signal") or {}
        outlook = slot.get("conditional_price_outlook")
        row: dict = {"time": target}
        if isinstance(outlook, dict):
            point = outlook["point"]
            side = signal["side"]
            color = UP_COLOR if side == "TOP" else DOWN_COLOR
            row.update(
                {
                    field: float(point[field])
                    for field in ("open", "high", "low", "close")
                }
            )
            row.update(
                {
                    "color": f"{color}66",
                    "borderColor": color,
                    "wickColor": color,
                }
            )
            close_row = {"time": target, "value": float(point["close"])}
            (top_closes if side == "TOP" else bottom_closes).append(close_row)
        data.append(row)
        metadata[target] = {
            "signal": signal,
            "probabilities": slot.get("probabilities"),
            "outlook": outlook,
        }
    return {
        "firstTime": str(forecast["slots"][0]["target_bucket"]),
        "data": data,
        "metadata": metadata,
        "topCloses": top_closes,
        "bottomCloses": bottom_closes,
    }


def _chart_payload(
    frame: pd.DataFrame,
    *,
    selected_smas: list[str],
    show_bollinger: bool,
    show_indicators: bool,
    technical_issuances: list[dict] | None,
    historical_signals: list[dict] | None,
    comparison_curve: list[dict] | None,
    validation_boundary: dict | None,
    technical_forecast: dict | None,
    initial_bars: int,
    initial_end_date: str | None,
) -> dict:
    candles = [
        {
            "time": _date(row.ts_event_utc),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        for row in frame[["ts_event_utc", "open", "high", "low", "close"]].itertuples(index=False)
    ]
    volume = [
        {
            "time": _date(row.ts_event_utc),
            "value": float(row.volume),
            "color": UP_COLOR if row.close >= row.open else DOWN_COLOR,
        }
        for row in frame[["ts_event_utc", "open", "close", "volume"]].itertuples(index=False)
    ]
    main_lines = [
        {
            "name": label,
            "color": SMA_COLORS[label],
            "data": _line(frame, f"sma_{label.removeprefix('SMA')}")
        }
        for label in selected_smas
    ]
    if show_bollinger:
        main_lines.extend(
            [
                {"name": "BOLL UPPER", "color": "#8B77BD", "data": _line(frame, "boll_upper"), "dashed": True},
                {"name": "BOLL MID", "color": "#6E5AA8", "data": _line(frame, "boll_mid"), "dashed": True},
                {"name": "BOLL LOWER", "color": "#8B77BD", "data": _line(frame, "boll_lower"), "dashed": True},
            ]
        )
    indicators = {}
    if show_indicators:
        indicators = {
            "macd_hist": _line(frame, "macd_hist"),
            "macd_dif": _line(frame, "macd_dif"),
            "macd_dea": _line(frame, "macd_dea"),
            "rsi": _line(frame, "rsi_14"),
            "kdj_k": _line(frame, "kdj_k"),
            "kdj_d": _line(frame, "kdj_d"),
            "kdj_j": _line(frame, "kdj_j"),
            "atr": _line(frame, "atr_14"),
            "volatility": _line(frame, "volatility_20"),
            "drawdown": _line(frame, "drawdown_from_high"),
        }
    initial_end_index = len(frame)
    if initial_end_date:
        eligible = frame.loc[
            frame["ts_event_utc"].dt.date
            <= pd.Timestamp(initial_end_date).date()
        ]
        if not eligible.empty:
            initial_end_index = int(eligible.index[-1]) + 1
    return {
        "candles": candles,
        "volume": volume,
        "mainLines": main_lines,
        "indicators": indicators,
        "markers": (
            _historical_markers(frame, historical_signals)
            if historical_signals is not None
            else _markers(frame, technical_issuances)
        ),
        "comparisonCurve": _comparison_curve(comparison_curve),
        "validationBoundary": validation_boundary,
        "futureForecast": _future_forecast(technical_forecast),
        "showIndicators": show_indicators,
        "initialBars": min(max(initial_bars, 20), len(frame)),
        "initialEndIndex": initial_end_index,
        "colors": {"up": UP_COLOR, "down": DOWN_COLOR, "grid": GRID_COLOR},
        "labels": {
            "open": tr("open"),
            "high": tr("high"),
            "low": tr("low"),
            "close": tr("close"),
            "disableWheelZoom": tr("disable_wheel_zoom"),
            "ohlc": tr("daily_candlestick"),
            "volume": tr("volume_chart"),
            "volatility": tr("volatility_20"),
            "drawdown": tr("drawdown_chart"),
            "performanceComparison": tr("validation_equity_comparison"),
            "modelStrategy": tr("validation_model_strategy"),
            "buyAndHold": tr("validation_buy_hold"),
        },
    }


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{margin:0;padding:0;background:#fcfdfb;color:#252824;font-family:Inter,"Segoe UI",sans-serif;letter-spacing:0;overflow:hidden}
#shell{width:100%;height:100%;display:flex;flex-direction:column}
#readout{height:52px;box-sizing:border-box;padding:5px 12px;border:1px solid #e2e5e0;border-bottom:0;background:linear-gradient(180deg,#fff,#f8f9f7);position:relative;font-size:12px;line-height:20px;overflow:hidden}
#readout-copy{min-width:0;overflow:hidden}
#ohlc{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#legend{display:flex;gap:12px;align-items:center;padding-right:154px;box-sizing:border-box;white-space:nowrap;overflow-x:auto;scrollbar-width:none;color:#636861}
#legend::-webkit-scrollbar{display:none}
.legend-item{display:inline-flex;align-items:center;gap:4px;flex:0 0 auto;font-size:11px}
.legend-swatch{width:10px;height:3px;border-radius:999px;display:inline-block}
#wheel-control{position:absolute;right:9px;bottom:5px;display:inline-flex;align-items:center;gap:5px;height:20px;box-sizing:border-box;padding:0 8px;border:1px solid #dde1da;border-radius:999px;background:#fff;color:#4f554d;font-size:10px;font-weight:600;line-height:1;white-space:nowrap;cursor:pointer;user-select:none;box-shadow:0 2px 8px rgba(20,22,19,.035);transition:border-color .18s ease,box-shadow .18s ease,transform .18s ease}
#wheel-control:hover{border-color:rgba(179,138,62,.42);box-shadow:0 5px 16px rgba(20,22,19,.07);transform:translateY(-1px)}
#wheel-lock{width:12px;height:12px;margin:0;accent-color:#8f6b29;cursor:pointer}
#chart{width:100%;flex:1;min-height:0;border:1px solid #e2e5e0;box-sizing:border-box;position:relative;background:#fcfdfb}
#future-band{position:absolute;top:0;bottom:0;display:none;pointer-events:none;z-index:8;border-left:2px dashed rgba(139,105,42,.48);background:linear-gradient(90deg,rgba(179,138,62,.035),rgba(179,138,62,.012))}
#validation-boundary{position:absolute;top:0;bottom:0;display:none;pointer-events:none;z-index:10;border-left:1px dashed rgba(116,91,42,.78)}
#validation-boundary span{position:absolute;top:7px;left:7px;padding:4px 7px;border:1px solid rgba(116,91,42,.16);border-radius:999px;background:rgba(255,255,255,.94);color:#765b28;font-size:9px;font-weight:700;line-height:1;white-space:nowrap;box-shadow:0 3px 12px rgba(20,22,19,.05)}
.pane-label{position:absolute;left:9px;top:6px;z-index:9;padding:3px 7px;border:1px solid rgba(32,35,31,.055);border-radius:999px;background:rgba(255,255,255,.88);color:#626760;font-size:10px;font-weight:600;pointer-events:none;letter-spacing:.02em;box-shadow:0 3px 12px rgba(20,22,19,.04)}
@media (prefers-reduced-motion:reduce){#wheel-control{transition:none}}
</style></head><body><div id="shell"><div id="readout"><div id="readout-copy"><span id="ohlc"></span><span id="legend"></span></div><label id="wheel-control"><input id="wheel-lock" type="checkbox" checked><span id="wheel-label"></span></label></div><div id="chart"></div></div>
<script>__LIBRARY__</script>
<script>
(()=>{
const payload=__PAYLOAD__;
const L=window.LightweightCharts;
const root=document.getElementById('chart');
const ohlc=document.getElementById('ohlc');
const legend=document.getElementById('legend');
const wheelLock=document.getElementById('wheel-lock');
document.getElementById('wheel-label').textContent=payload.labels.disableWheelZoom;
const chart=L.createChart(root,{
  width:Math.max(320,Math.floor(root.clientWidth)),height:__CHART_HEIGHT__,
  layout:{background:{type:'solid',color:'#fcfdfb'},textColor:'#626760',fontSize:11,fontFamily:'Inter,"Segoe UI",sans-serif',panes:{enableResize:false,separatorColor:'#e2e5e0',separatorHoverColor:'#d8dcd6'},attributionLogo:true},
  grid:{vertLines:{color:'#f0f2ef'},horzLines:{color:'#f0f2ef'}},
  leftPriceScale:{visible:false,autoScale:true,borderColor:'#d9ddd7'},
  rightPriceScale:{visible:true,autoScale:true,borderColor:'#d9ddd7',scaleMargins:{top:.09,bottom:.09}},
  timeScale:{borderColor:'#d9ddd7',rightOffset:3,barSpacing:7,minBarSpacing:.5,fixLeftEdge:false,fixRightEdge:false,lockVisibleTimeRangeOnResize:true,timeVisible:false,secondsVisible:false},
  crosshair:{mode:L.CrosshairMode.Normal,vertLine:{color:'#939a91',width:1,style:2,labelBackgroundColor:'#52584f'},horzLine:{color:'#939a91',width:1,style:2,labelBackgroundColor:'#52584f'}},
  handleScroll:{mouseWheel:false,pressedMouseMove:true,horzTouchDrag:true,vertTouchDrag:false},
  handleScale:{mouseWheel:!wheelLock.checked,pinch:true,axisPressedMouseMove:{time:true,price:false},axisDoubleClickReset:{time:true,price:true}},
  kineticScroll:{touch:true,mouse:true}
});
wheelLock.addEventListener('change',()=>chart.applyOptions({
  handleScale:{mouseWheel:!wheelLock.checked,pinch:true,axisPressedMouseMove:{time:true,price:false},axisDoubleClickReset:{time:true,price:true}}
}));
const basePaneCount=payload.showIndicators?8:2;
const comparisonPane=basePaneCount;
const paneCount=basePaneCount+(payload.comparisonCurve?1:0);
while(chart.panes().length<paneCount)chart.addPane();
const candle=chart.addSeries(L.CandlestickSeries,{upColor:payload.colors.up,downColor:payload.colors.down,borderUpColor:payload.colors.up,borderDownColor:payload.colors.down,wickUpColor:payload.colors.up,wickDownColor:payload.colors.down,priceLineVisible:false,lastValueVisible:true},0);
candle.setData(payload.candles);
let future=null;
let futureBand=null;
let validationBoundary=null;
if(payload.validationBoundary){
  validationBoundary=document.createElement('div');
  validationBoundary.id='validation-boundary';
  const label=document.createElement('span');
  label.textContent=payload.validationBoundary.label;
  validationBoundary.appendChild(label);
  root.appendChild(validationBoundary);
}
if(payload.futureForecast){
  future=chart.addSeries(L.CandlestickSeries,{upColor:'rgba(200,75,75,.35)',downColor:'rgba(35,139,87,.35)',borderUpColor:payload.colors.up,borderDownColor:payload.colors.down,wickUpColor:payload.colors.up,wickDownColor:payload.colors.down,priceLineVisible:false,lastValueVisible:false},0);
  future.setData(payload.futureForecast.data);
  const futureMarkers=payload.futureForecast.data.filter(row=>row.open!==undefined).map(row=>{const meta=payload.futureForecast.metadata[row.time];const side=meta.signal.side;return {time:row.time,position:side==='TOP'?'aboveBar':'belowBar',shape:side==='TOP'?'arrowDown':'arrowUp',color:side==='TOP'?payload.colors.up:payload.colors.down,text:`${side} ${meta.signal.strength}`};});
  if(futureMarkers.length)L.createSeriesMarkers(future,futureMarkers,{autoScale:true});
  if(payload.futureForecast.topCloses.length)addLine(payload.futureForecast.topCloses,payload.colors.up,0,'TOP conditional close',{width:2,dashed:true});
  if(payload.futureForecast.bottomCloses.length)addLine(payload.futureForecast.bottomCloses,payload.colors.down,0,'BOTTOM conditional close',{width:2,dashed:true});
  futureBand=document.createElement('div');futureBand.id='future-band';root.appendChild(futureBand);
}
for(const line of payload.mainLines){const series=chart.addSeries(L.LineSeries,{title:'',color:line.color,lineWidth:line.dashed?1:2,lineStyle:line.dashed?L.LineStyle.Dashed:L.LineStyle.Solid,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false},0);series.setData(line.data);const item=document.createElement('span');item.className='legend-item';const swatch=document.createElement('i');swatch.className='legend-swatch';swatch.style.background=line.color;item.appendChild(swatch);item.appendChild(document.createTextNode(line.name));legend.appendChild(item)}
const volume=chart.addSeries(L.HistogramSeries,{priceFormat:{type:'volume'},priceLineVisible:false,lastValueVisible:false},1);volume.setData(payload.volume);
if(payload.markers.length)L.createSeriesMarkers(candle,payload.markers,{autoScale:true});
function addLine(data,color,pane,title,options={}){const s=chart.addSeries(L.LineSeries,{title,color,lineWidth:options.width||1,lineStyle:options.dashed?L.LineStyle.Dashed:L.LineStyle.Solid,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false,priceFormat:options.percent?{type:'custom',minMove:.0001,formatter:v=>(v*100).toFixed(2)+'%'}:undefined},pane);s.setData(data);return s}
function addRule(series,price,color){series.createPriceLine({price,color,lineWidth:1,lineStyle:L.LineStyle.Dashed,axisLabelVisible:false,title:''})}
if(payload.showIndicators){
  const mh=chart.addSeries(L.HistogramSeries,{priceLineVisible:false,lastValueVisible:false},2);mh.setData(payload.indicators.macd_hist.map(x=>({...x,color:x.value>=0?payload.colors.up:payload.colors.down})));
  addLine(payload.indicators.macd_dif,'#2F6B9A',2,'DIF');addLine(payload.indicators.macd_dea,'#D1872C',2,'DEA');
  const rsi=addLine(payload.indicators.rsi,'#2F6B9A',3,'RSI(14)',{width:2});addRule(rsi,30,'#D1872C');addRule(rsi,70,'#D1872C');
  const k=addLine(payload.indicators.kdj_k,'#2F6B9A',4,'K');addLine(payload.indicators.kdj_d,'#D1872C',4,'D');addLine(payload.indicators.kdj_j,'#6E5AA8',4,'J');addRule(k,20,'#D1872C');addRule(k,80,'#D1872C');
  addLine(payload.indicators.atr,'#D1872C',5,'ATR(14)',{width:2});
  addLine(payload.indicators.volatility,'#278A8A',6,payload.labels.volatility,{percent:true,width:2});
  const dd=addLine(payload.indicators.drawdown,'#8B5D3B',7,payload.labels.drawdown,{percent:true,width:2});addRule(dd,0,'#aab1b6');
}
if(payload.comparisonCurve){
  const modelSeries=addLine(payload.comparisonCurve.model,'#A47724',comparisonPane,payload.labels.modelStrategy,{percent:true,width:2});
  addLine(payload.comparisonCurve.buyHold,'#596879',comparisonPane,payload.labels.buyAndHold,{percent:true,width:2});
  addRule(modelSeries,0,'#aab1b6');
}
const factors=payload.showIndicators?[4,1,1.15,1,1,1,1,1]:[4,1];
const paneLabels=payload.showIndicators?[payload.labels.ohlc,payload.labels.volume,'MACD (12, 26, 9)','RSI (14)','KDJ (9, 3, 3)','ATR (14)',payload.labels.volatility,payload.labels.drawdown]:[payload.labels.ohlc,payload.labels.volume];
if(payload.comparisonCurve){factors.push(1.6);paneLabels.push(payload.labels.performanceComparison);}
chart.panes().forEach((pane,index)=>pane.setStretchFactor(factors[index]||1));
requestAnimationFrame(()=>chart.panes().forEach((pane,index)=>{const host=pane.getHTMLElement();if(!host)return;host.style.position='relative';const label=document.createElement('div');label.className='pane-label';label.textContent=paneLabels[index];host.appendChild(label)}));
function showBar(bar,time){if(!bar)return;ohlc.textContent=`${time||''}   ${payload.labels.open} ${bar.open.toFixed(2)}   ${payload.labels.high} ${bar.high.toFixed(2)}   ${payload.labels.low} ${bar.low.toFixed(2)}   ${payload.labels.close} ${bar.close.toFixed(2)}`}
const initialReadoutIndex=Math.max(0,payload.initialEndIndex-1);
showBar(payload.candles[initialReadoutIndex],payload.candles[initialReadoutIndex]?.time);
function timeKey(time){return typeof time==='string'?time:(time?`${time.year}-${String(time.month).padStart(2,'0')}-${String(time.day).padStart(2,'0')}`:'')}
chart.subscribeCrosshairMove(param=>{const key=timeKey(param.time);const futureMeta=payload.futureForecast?.metadata?.[key];if(futureMeta){const signal=futureMeta.signal;const probs=futureMeta.probabilities||{};if(signal.side==='NORMAL'){ohlc.textContent=`${key}   NORMAL   No turning-point action signal; conditional price hidden`;}else{const point=futureMeta.outlook?.point;const interval=futureMeta.outlook?.marginal_80pct_intervals?.close;ohlc.textContent=`${key}   ${signal.side} ${signal.strength}   NORMAL ${(100*(probs.NORMAL||0)).toFixed(2)}% TOP ${(100*((probs.TOP_L1||0)+(probs.TOP_L2||0)+(probs.TOP_L3||0))).toFixed(2)}% BOTTOM ${(100*((probs.BOTTOM_L1||0)+(probs.BOTTOM_L2||0)+(probs.BOTTOM_L3||0))).toFixed(2)}%${point?`   O ${point.open.toFixed(2)} H ${point.high.toFixed(2)} L ${point.low.toFixed(2)} C ${point.close.toFixed(2)}`:''}${interval?`   close marginal 80% [${interval.lower.toFixed(2)}, ${interval.upper.toFixed(2)}]`:''}   Advisory only; does not control trading`;}}else{const bar=param.seriesData.get(candle);if(bar)showBar(bar,key)}});
function updateFutureBand(){if(!futureBand||!payload.futureForecast)return;const coordinate=chart.timeScale().timeToCoordinate(payload.futureForecast.firstTime);if(coordinate===null){futureBand.style.display='none';return;}futureBand.style.display='block';futureBand.style.left=`${Math.max(0,coordinate-4)}px`;futureBand.style.right='0';}
function updateValidationBoundary(){if(!validationBoundary||!payload.validationBoundary)return;const coordinate=chart.timeScale().timeToCoordinate(payload.validationBoundary.date);if(coordinate===null){validationBoundary.style.display='none';return;}validationBoundary.style.display='block';validationBoundary.style.left=`${Math.max(0,coordinate)}px`;const label=validationBoundary.firstElementChild;if(label){const placeLeft=coordinate+label.offsetWidth+14>root.clientWidth;label.style.left=placeLeft?'auto':'7px';label.style.right=placeLeft?'7px':'auto';}}
function updateOverlays(){updateFutureBand();updateValidationBoundary();}
chart.timeScale().subscribeVisibleLogicalRangeChange(updateOverlays);requestAnimationFrame(updateOverlays);
const initialEnd=payload.initialEndIndex;chart.timeScale().setVisibleLogicalRange({from:Math.max(-.5,initialEnd-payload.initialBars-.5),to:initialEnd+2});
const observer=new ResizeObserver(entries=>{const width=Math.floor(entries[0].contentRect.width);if(width>0){chart.applyOptions({width});requestAnimationFrame(updateOverlays)}});observer.observe(root);
})();
</script></body></html>"""


def render_gold_daily_chart(
    payload: dict,
    *,
    key_prefix: str,
    technical_issuances: list[dict] | None = None,
    historical_signals: list[dict] | None = None,
    comparison_curve: list[dict] | None = None,
    validation_boundary: dict | None = None,
    technical_forecast: dict | None = None,
    initial_bars: int = 180,
    initial_end_date: str | None = None,
    show_indicators: bool = True,
) -> pd.DataFrame:
    """Render a shared, x-only financial chart whose Y axes auto-fit visible bars."""
    frame = market_frame(payload)
    safe_key = re.sub(r"[^A-Za-z0-9_]", "_", key_prefix)
    selected_smas = st.multiselect(
        tr("moving_averages"),
        options=list(SMA_COLORS),
        default=["SMA5", "SMA10", "SMA20"],
        key=f"{safe_key}_sma_selection",
    )
    show_bollinger = st.checkbox(tr("bollinger_band"), value=True, key=f"{safe_key}_bollinger")
    chart_payload = _chart_payload(
        frame,
        selected_smas=selected_smas,
        show_bollinger=show_bollinger,
        show_indicators=show_indicators,
        technical_issuances=technical_issuances,
        historical_signals=historical_signals,
        comparison_curve=comparison_curve,
        validation_boundary=validation_boundary,
        technical_forecast=technical_forecast,
        initial_bars=initial_bars,
        initial_end_date=initial_end_date,
    )
    payload_json = json.dumps(
        chart_payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).replace("<", "\\u003c")
    chart_height = (
        1360
        if show_indicators and comparison_curve
        else 1180
        if show_indicators
        else 820
        if comparison_curve
        else 650
    )
    html = (
        _HTML.replace("__LIBRARY__", _library_source())
        .replace("__PAYLOAD__", payload_json)
        .replace("__CHART_HEIGHT__", str(chart_height))
    )
    st.iframe(html, height=chart_height + 54, tab_index=0)
    return frame
