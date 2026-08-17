function List({ items, emptyLabel }) {
  if (!items || items.length === 0) {
    return <p className="rr-empty">{emptyLabel}</p>;
  }
  return (
    <ul className="rr-list">
      {items.map((item, i) => (
        <li key={i}>{item}</li>
      ))}
    </ul>
  );
}

export default function ReasonsRisks({ reasons, risks }) {
  return (
    <div className="rr-grid">
      <div>
        <p className="rr-col__title rr-col__title--reasons">Top reasons</p>
        <List items={reasons} emptyLabel="No SHAP-backed reasons available for this symbol." />
      </div>
      <div>
        <p className="rr-col__title rr-col__title--risks">Risk factors</p>
        <List items={risks} emptyLabel="No elevated risk factors flagged." />
      </div>
    </div>
  );
}
