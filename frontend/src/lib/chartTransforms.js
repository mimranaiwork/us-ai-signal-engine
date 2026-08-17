// Renko and Point & Figure are price-based (not time-based) chart
// constructions — a fixed box/brick size, direction determined purely by
// how much price has moved, not by trading sessions. These are honest,
// standard simplified implementations (no double-top/bottom pattern
// annotation for P&F, no separate reversal-brick convention for Renko) —
// real constructions, just not every charting-platform nuance.

// Brick size for Renko / box size for Point & Figure, both ATR-based: a
// volatility-scaled size so the brick/box count stays reasonable across
// very different price levels and volatility regimes. Not a
// backtested/validated parameter, same honesty class as
// features/technical/pivot_points.py's BREAKOUT_PROXIMITY_PCT.
export function defaultBoxSize(bars) {
  const atrs = bars.map((b) => b.atr_14).filter((v) => v != null && v > 0);
  if (atrs.length) return atrs[atrs.length - 1];
  const last = bars[bars.length - 1]?.adj_close || 1;
  return last * 0.01;
}

export function toRenkoBricks(bars, brickSize) {
  if (!brickSize || brickSize <= 0 || bars.length === 0) return [];
  const bricks = [];
  let base = bars[0].adj_close;
  for (const b of bars) {
    const price = b.adj_close;
    while (price - base >= brickSize) {
      const open = base;
      base += brickSize;
      bricks.push({ open, close: base, direction: "up", date: b.date });
    }
    while (base - price >= brickSize) {
      const open = base;
      base -= brickSize;
      bricks.push({ open, close: base, direction: "down", date: b.date });
    }
  }
  return bricks;
}

// Classic column construction: extend the current column while price keeps
// moving the same direction; start a new column once price reverses by at
// least `reversal` boxes.
export function toPointFigureColumns(bars, boxSize, reversal = 3) {
  if (!boxSize || boxSize <= 0 || bars.length === 0) return [];
  const toBox = (price) => Math.round(price / boxSize);

  const columns = [];
  let direction = null; // 1 = X column (up), -1 = O column (down)

  for (const b of bars) {
    const box = toBox(b.adj_close);
    if (direction === null) {
      direction = 1;
      columns.push({ direction, top: box, bottom: box, date: b.date });
      continue;
    }
    const col = columns[columns.length - 1];
    if (direction === 1) {
      if (box > col.top) {
        col.top = box;
        col.date = b.date;
      } else if (col.top - box >= reversal) {
        direction = -1;
        columns.push({ direction, top: col.top - 1, bottom: box, date: b.date });
      }
    } else {
      if (box < col.bottom) {
        col.bottom = box;
        col.date = b.date;
      } else if (box - col.bottom >= reversal) {
        direction = 1;
        columns.push({ direction, top: box, bottom: col.bottom + 1, date: b.date });
      }
    }
  }

  return columns.map((c) => ({ ...c, topPrice: c.top * boxSize, bottomPrice: c.bottom * boxSize }));
}
