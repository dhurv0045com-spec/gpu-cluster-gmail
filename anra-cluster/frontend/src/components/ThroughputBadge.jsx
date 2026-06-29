export default function ThroughputBadge({ tokensPerSecond, label = 'Throughput' }) {
  const value = tokensPerSecond || 0
  const display = value >= 1000 ? `${(value / 1000).toFixed(1)}k` : value.toFixed(0)

  return (
    <div className="inline-flex items-center gap-2 bg-[#0d1225] border border-gray-800 rounded-lg px-3 py-2 font-mono">
      <span className="text-xs text-gray-600">{label}</span>
      <span className="text-accent text-sm font-bold">{display}</span>
      <span className="text-xs text-gray-500">tok/s</span>
    </div>
  )
}
