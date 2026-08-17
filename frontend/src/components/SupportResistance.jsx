function Level({ label, price, kind, isNearest }) {
  return (
    <div className={`sr-level sr-level--${kind} ${isNearest ? "sr-level--nearest" : ""}`}>
      <span className="sr-level__label">{label}</span>
      <span className="sr-level__price">{price.toFixed(2)}</span>
    </div>
  );
}

export default function SupportResistance({ levels }) {
  if (!levels || !levels.available) {
    return (
      <>
        <p className="section-label">Support &amp; resistance</p>
        <p className="rr-empty">Not enough price history yet to compute pivot levels.</p>
      </>
    );
  }

  const { current_price, resistance, support, nearest_resistance, nearest_support, near_breakout, target, expected_profit_pct } = levels;

  return (
    <>
      <div className="sr-header">
        <p className="section-label" style={{ margin: 0 }}>
          Support &amp; resistance
        </p>
        {near_breakout && <span className="tag tag--breakout">Near breakout</span>}
      </div>

      <div className="sr-ladder">
        <Level label="R3" price={resistance.r3} kind="resistance" isNearest={resistance.r3 === nearest_resistance} />
        <Level label="R2" price={resistance.r2} kind="resistance" isNearest={resistance.r2 === nearest_resistance} />
        <Level label="R1" price={resistance.r1} kind="resistance" isNearest={resistance.r1 === nearest_resistance} />
        <div className="sr-level sr-level--current">
          <span className="sr-level__label">PRICE</span>
          <span className="sr-level__price">{current_price.toFixed(2)}</span>
        </div>
        <Level label="S1" price={support.s1} kind="support" isNearest={support.s1 === nearest_support} />
        <Level label="S2" price={support.s2} kind="support" isNearest={support.s2 === nearest_support} />
        <Level label="S3" price={support.s3} kind="support" isNearest={support.s3 === nearest_support} />
      </div>

      {target != null && (
        <p className="sr-target">
          Target (nearest resistance): <strong>{target.toFixed(2)}</strong>
          {expected_profit_pct != null && (
            <span className={expected_profit_pct >= 0 ? "sr-target__profit--pos" : "sr-target__profit--neg"}>
              {" "}
              ({expected_profit_pct >= 0 ? "+" : ""}
              {expected_profit_pct.toFixed(2)}%)
            </span>
          )}
        </p>
      )}
      {target == null && <p className="rr-empty">Price is already above all three resistance levels.</p>}

      <p className="footer-note">
        Classic daily pivot points from the prior session's high/low/close — a geometric read of price vs. levels, not a
        model prediction.
      </p>
    </>
  );
}
