import { useEffect, useMemo, useRef, useState } from "react";
import { fetchPrices } from "../api.js";
import { defaultBoxSize, toPointFigureColumns, toRenkoBricks } from "../lib/chartTransforms.js";

const WIDTH = 900;
const PAD = { top: 10, right: 74, bottom: 6, left: 4 };
const PRICE_H = 240;
const VOLUME_H = 60;
const RSI_H = 80;
const GAP = 6;

const CHART_TYPES = ["Candlestick", "Line", "Renko", "Point & Figure"];

// `get` reads the value out of the signal endpoint's `levels` shape:
// {pivot, resistance: {r1,r2,r3}, support: {s1,s2,s3}}. `tier` fades R2/R3
// and S2/S3 relative to the nearest, most actionable R1/S1/pivot — without
// it, three same-weight dashed lines this close together render as one
// solid stripe instead of three distinguishable levels.
const LEVEL_LINES = [
  { key: "r3", label: "R3", cls: "chart-level--resistance", tier: 3, get: (lv) => lv.resistance.r3 },
  { key: "r2", label: "R2", cls: "chart-level--resistance", tier: 2, get: (lv) => lv.resistance.r2 },
  { key: "r1", label: "R1", cls: "chart-level--resistance", tier: 1, get: (lv) => lv.resistance.r1 },
  { key: "pivot", label: "P", cls: "chart-level--pivot", tier: 1, get: (lv) => lv.pivot },
  { key: "s1", label: "S1", cls: "chart-level--support", tier: 1, get: (lv) => lv.support.s1 },
  { key: "s2", label: "S2", cls: "chart-level--support", tier: 2, get: (lv) => lv.support.s2 },
  { key: "s3", label: "S3", cls: "chart-level--support", tier: 3, get: (lv) => lv.support.s3 },
];
const MIN_LABEL_GAP = 13;

function declutterLabelPositions(items, top, bottom) {
  const labelY = items.map((it) => it.yPos);
  for (let i = 1; i < labelY.length; i++) {
    if (labelY[i] - labelY[i - 1] < MIN_LABEL_GAP) labelY[i] = labelY[i - 1] + MIN_LABEL_GAP;
  }
  const overflow = labelY[labelY.length - 1] - bottom;
  if (overflow > 0) {
    for (let i = labelY.length - 1; i >= 0; i--) labelY[i] -= overflow;
  }
  if (labelY[0] < top) labelY[0] = top;
  for (let i = 1; i < labelY.length; i++) {
    if (labelY[i] - labelY[i - 1] < MIN_LABEL_GAP) labelY[i] = labelY[i - 1] + MIN_LABEL_GAP;
  }
  return items.map((it, i) => ({ ...it, labelY: labelY[i] }));
}

export default function PriceChart({ symbol, levels }) {
  const [bars, setBars] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [hoverIdx, setHoverIdx] = useState(null);
  const [chartType, setChartType] = useState("Candlestick");
  const svgRef = useRef(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setBars(null);
    fetchPrices(symbol, 120)
      .then((res) => setBars(res.bars.filter((b) => b.adj_close != null)))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [symbol]);

  const plot = useMemo(() => {
    if (!bars || bars.length < 2) return null;

    const plotW = WIDTH - PAD.left - PAD.right;
    const priceTop = PAD.top;
    const priceBottom = PAD.top + PRICE_H;
    const volTop = priceBottom + GAP;
    const volBottom = volTop + VOLUME_H;
    const rsiTop = volBottom + GAP;
    const rsiBottom = rsiTop + RSI_H;
    const totalHeight = rsiBottom + PAD.bottom;

    const currentLevels = levels && levels.available ? levels : null;
    const showOverlays = chartType === "Candlestick" || chartType === "Line";

    // ── Price series (candlestick/line share the same bar-indexed x scale) ──
    let renko = null, pointFigure = null;
    let lo, hi, xCount;

    if (chartType === "Renko") {
      const brickSize = defaultBoxSize(bars);
      renko = toRenkoBricks(bars, brickSize);
      xCount = Math.max(renko.length, 1);
      const vals = renko.length ? renko.flatMap((b) => [b.open, b.close]) : bars.map((b) => b.adj_close);
      lo = Math.min(...vals);
      hi = Math.max(...vals);
    } else if (chartType === "Point & Figure") {
      const boxSize = defaultBoxSize(bars) * 0.6;
      pointFigure = toPointFigureColumns(bars, boxSize, 3);
      xCount = Math.max(pointFigure.length, 1);
      const vals = pointFigure.length ? pointFigure.flatMap((c) => [c.topPrice, c.bottomPrice]) : bars.map((b) => b.adj_close);
      lo = Math.min(...vals);
      hi = Math.max(...vals);
    } else {
      xCount = bars.length;
      const highs = chartType === "Candlestick" ? bars.map((b) => b.high ?? b.adj_close) : bars.map((b) => b.adj_close);
      const lows = chartType === "Candlestick" ? bars.map((b) => b.low ?? b.adj_close) : bars.map((b) => b.adj_close);
      lo = Math.min(...lows);
      hi = Math.max(...highs);
      if (showOverlays) {
        const bbVals = bars.flatMap((b) => [b.bollinger_upper, b.bollinger_lower]).filter((v) => v != null);
        if (bbVals.length) {
          lo = Math.min(lo, ...bbVals);
          hi = Math.max(hi, ...bbVals);
        }
      }
    }

    if (showOverlays && currentLevels) {
      const levelValues = [
        currentLevels.resistance.r1, currentLevels.resistance.r2, currentLevels.resistance.r3,
        currentLevels.support.s1, currentLevels.support.s2, currentLevels.support.s3,
      ];
      lo = Math.min(lo, ...levelValues);
      hi = Math.max(hi, ...levelValues);
    }
    const margin = (hi - lo) * 0.06 || hi * 0.02;
    lo -= margin;
    hi += margin;

    const x = (i) => PAD.left + (xCount === 1 ? plotW / 2 : (i / (xCount - 1)) * plotW);
    const barW = Math.max(1.5, Math.min(9, (plotW / xCount) * 0.6));
    const y = (v) => priceTop + PRICE_H - ((v - lo) / (hi - lo)) * PRICE_H;

    let linePath = null, bbBandPath = null, bbMidPath = null, bbUpperPath = null, bbLowerPath = null, bbLastUpperY = null;
    if (chartType === "Line") {
      linePath = bars.map((b, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(b.adj_close).toFixed(1)}`).join(" ");
    }
    if (showOverlays) {
      const withBB = bars.map((b, i) => ({ i, u: b.bollinger_upper, l: b.bollinger_lower, m: b.bollinger_mid })).filter((r) => r.u != null && r.l != null);
      if (withBB.length > 1) {
        bbUpperPath = withBB.map((r) => `${r.i === withBB[0].i ? "M" : "L"}${x(r.i).toFixed(1)},${y(r.u).toFixed(1)}`).join(" ");
        bbLowerPath = withBB.map((r) => `${r.i === withBB[0].i ? "M" : "L"}${x(r.i).toFixed(1)},${y(r.l).toFixed(1)}`).join(" ");
        const lowerRev = [...withBB].reverse().map((r) => `L${x(r.i).toFixed(1)},${y(r.l).toFixed(1)}`).join(" ");
        bbBandPath = `${bbUpperPath} ${lowerRev} Z`;
        bbMidPath = withBB.map((r) => `${r.i === withBB[0].i ? "M" : "L"}${x(r.i).toFixed(1)},${y(r.m).toFixed(1)}`).join(" ");
        bbLastUpperY = y(withBB[withBB.length - 1].u);
      }
    }

    let levelLines = showOverlays && currentLevels
      ? LEVEL_LINES.map((l) => ({ ...l, value: l.get(currentLevels), yPos: y(l.get(currentLevels)) }))
      : [];
    if (levelLines.length) levelLines = declutterLabelPositions(levelLines, priceTop, priceBottom);

    // ── Volume subplot ──
    const volumes = bars.map((b) => b.volume || 0);
    const maxVol = Math.max(...volumes, 1);
    const volY = (v) => volBottom - (v / maxVol) * VOLUME_H;

    // ── RSI subplot ──
    const rsiY = (v) => rsiBottom - (v / 100) * RSI_H;
    const rsiPoints = bars.map((b, i) => ({ i, v: b.rsi_14 })).filter((r) => r.v != null);
    const rsiPath = rsiPoints.length > 1
      ? rsiPoints.map((r, idx) => `${idx === 0 ? "M" : "L"}${x(r.i).toFixed(1)},${rsiY(r.v).toFixed(1)}`).join(" ")
      : null;

    return {
      x, y, barW, totalHeight, priceTop, priceBottom, volTop, volBottom, rsiTop, rsiBottom,
      renko, pointFigure, linePath, bbBandPath, bbMidPath, bbUpperPath, bbLowerPath, bbLastUpperY, levelLines,
      volumes, volY, rsiPath, rsiY, xCount,
    };
  }, [bars, levels, chartType]);

  const handleMouseMove = (e) => {
    if (!bars || !plot) return;
    const rect = svgRef.current.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * WIDTH;
    const frac = (relX - PAD.left) / (WIDTH - PAD.left - PAD.right);
    const idx = Math.round(frac * (plot.xCount - 1));
    setHoverIdx(Math.max(0, Math.min(plot.xCount - 1, idx)));
  };

  if (loading) return <p className="state-message">Loading chart...</p>;
  if (error) return <p className="rr-empty">Chart unavailable: {error}</p>;
  if (!plot) return <p className="rr-empty">Not enough price history for a chart.</p>;

  const hoveredBar = hoverIdx != null && (chartType === "Candlestick" || chartType === "Line") ? bars[hoverIdx] : bars[bars.length - 1];

  return (
    <div className="price-chart">
      <div className="price-chart__toolbar">
        {CHART_TYPES.map((t) => (
          <button
            key={t}
            className={`filter-chip ${chartType === t ? "filter-chip--active" : ""}`}
            onClick={() => setChartType(t)}
          >
            {t}
          </button>
        ))}
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${plot.totalHeight}`}
        className="price-chart__svg"
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoverIdx(null)}
      >
        {/* Bollinger Bands overlay */}
        {plot.bbBandPath && <path d={plot.bbBandPath} className="price-chart__bb-band" />}
        {plot.bbUpperPath && <path d={plot.bbUpperPath} className="price-chart__bb-edge" />}
        {plot.bbLowerPath && <path d={plot.bbLowerPath} className="price-chart__bb-edge" />}
        {plot.bbMidPath && <path d={plot.bbMidPath} className="price-chart__bb-mid" />}
        {plot.bbLastUpperY != null && (
          <text x={WIDTH - PAD.right - 4} y={plot.bbLastUpperY - 4} className="price-chart__bb-label" textAnchor="end">
            Bollinger (20, 2)
          </text>
        )}

        {/* Main price panel */}
        {chartType === "Line" && plot.linePath && <path d={plot.linePath} className="price-chart__line" />}

        {chartType === "Candlestick" &&
          bars.map((b, i) => {
            const up = b.close >= b.open;
            const cx = plot.x(i);
            return (
              <g key={i}>
                <line x1={cx} x2={cx} y1={plot.y(b.high ?? b.adj_close)} y2={plot.y(b.low ?? b.adj_close)} className={up ? "candle-wick--up" : "candle-wick--down"} />
                <rect
                  x={cx - plot.barW / 2}
                  y={plot.y(Math.max(b.open, b.close))}
                  width={plot.barW}
                  height={Math.max(1, Math.abs(plot.y(b.open) - plot.y(b.close)))}
                  className={up ? "candle-body--up" : "candle-body--down"}
                />
              </g>
            );
          })}

        {chartType === "Renko" &&
          plot.renko.map((b, i) => (
            <rect
              key={i}
              x={plot.x(i) - plot.barW / 2}
              y={plot.y(Math.max(b.open, b.close))}
              width={plot.barW}
              height={Math.max(1, Math.abs(plot.y(b.open) - plot.y(b.close)))}
              className={b.direction === "up" ? "candle-body--up" : "candle-body--down"}
            />
          ))}

        {chartType === "Point & Figure" &&
          plot.pointFigure.map((c, i) => (
            <rect
              key={i}
              x={plot.x(i) - plot.barW / 2}
              y={plot.y(c.topPrice)}
              width={plot.barW}
              height={Math.max(1, plot.y(c.bottomPrice) - plot.y(c.topPrice))}
              className={c.direction === 1 ? "pf-column--x" : "pf-column--o"}
            />
          ))}

        {/* S/R pivot levels — tier 1 (nearest, most actionable) full weight, tier 2/3 fade out so three close-together levels don't blend into one stripe */}
        {plot.levelLines.map((l) => (
          <g key={l.key} className={`chart-level-tier--${l.tier}`}>
            <line x1={PAD.left} x2={WIDTH - PAD.right} y1={l.yPos} y2={l.yPos} className={`chart-level ${l.cls}`} />
            {Math.abs(l.labelY - l.yPos) > 1 && (
              <line x1={WIDTH - PAD.right} x2={WIDTH - PAD.right + 5} y1={l.yPos} y2={l.labelY} className={`chart-level-leader ${l.cls}`} />
            )}
            <text x={WIDTH - PAD.right + 7} y={l.labelY + 4} className={`chart-level-label ${l.cls}`}>
              {l.label} {l.value.toFixed(2)}
            </text>
          </g>
        ))}

        {/* Volume subplot */}
        <line x1={PAD.left} x2={WIDTH - PAD.right} y1={plot.volTop} y2={plot.volTop} className="panel-divider" />
        <text x={PAD.left} y={plot.volTop + 10} className="panel-label">VOLUME</text>
        {bars.map((b, i) => {
          const up = (b.close ?? b.adj_close) >= (b.open ?? b.adj_close);
          return (
            <rect
              key={i}
              x={plot.x(i) - plot.barW / 2}
              y={plot.volY(b.volume || 0)}
              width={plot.barW}
              height={Math.max(0.5, plot.volBottom - plot.volY(b.volume || 0))}
              className={up ? "volume-bar--up" : "volume-bar--down"}
            />
          );
        })}

        {/* RSI subplot */}
        <line x1={PAD.left} x2={WIDTH - PAD.right} y1={plot.rsiTop} y2={plot.rsiTop} className="panel-divider" />
        <text x={PAD.left} y={plot.rsiTop + 10} className="panel-label">RSI (14)</text>
        <line x1={PAD.left} x2={WIDTH - PAD.right} y1={plot.rsiY(70)} y2={plot.rsiY(70)} className="rsi-ref-line" />
        <line x1={PAD.left} x2={WIDTH - PAD.right} y1={plot.rsiY(50)} y2={plot.rsiY(50)} className="rsi-ref-line rsi-ref-line--mid" />
        <line x1={PAD.left} x2={WIDTH - PAD.right} y1={plot.rsiY(30)} y2={plot.rsiY(30)} className="rsi-ref-line" />
        <text x={WIDTH - PAD.right + 5} y={plot.rsiY(70) + 3} className="rsi-ref-label">70</text>
        <text x={WIDTH - PAD.right + 5} y={plot.rsiY(30) + 3} className="rsi-ref-label">30</text>
        {plot.rsiPath && <path d={plot.rsiPath} className="price-chart__rsi-line" />}

        {hoverIdx != null && (
          <line x1={plot.x(hoverIdx)} x2={plot.x(hoverIdx)} y1={PAD.top} y2={plot.rsiBottom} className="price-chart__hover-line" />
        )}
      </svg>

      <div className="price-chart__footer">
        <span>{hoveredBar.date}</span>
        {(chartType === "Candlestick") && hoveredBar.open != null ? (
          <span className="price-chart__ohlc">
            O <strong>{hoveredBar.open?.toFixed(2)}</strong> H <strong>{hoveredBar.high?.toFixed(2)}</strong> L{" "}
            <strong>{hoveredBar.low?.toFixed(2)}</strong> C <strong>{hoveredBar.close?.toFixed(2)}</strong>
          </span>
        ) : (
          <span className="price-chart__hover-price">{hoveredBar.adj_close?.toFixed(2)}</span>
        )}
        {hoveredBar.rsi_14 != null && <span className="footer-note">RSI {hoveredBar.rsi_14.toFixed(1)}</span>}
        <span className="footer-note" style={{ marginLeft: "auto" }}>
          Last {bars.length} sessions
        </span>
      </div>
    </div>
  );
}
