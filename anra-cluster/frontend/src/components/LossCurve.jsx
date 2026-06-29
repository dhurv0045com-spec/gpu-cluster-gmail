import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid } from 'recharts'

export default function LossCurve({ data = [] }) {
  const chartData = data.map((v, i) => ({ step: i, loss: typeof v === 'number' ? v : 0 }))
  const isEmpty = chartData.length === 0 || chartData.every(d => d.loss === 0)

  return (
    <div className="bg-[#0d1225] border border-gray-800 rounded-lg p-4">
      <h3 className="text-xs font-mono text-gray-600 uppercase tracking-wider mb-3">Loss Curve</h3>
      {isEmpty ? (
        <div className="flex items-center justify-center h-[160px] text-gray-700 font-mono text-xs">
          Waiting for training data...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a2040" />
            <XAxis dataKey="step" tick={{ fill: '#4b5563', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis
              domain={['dataMin - 0.1', 'dataMax + 0.1']}
              tick={{ fill: '#4b5563', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0d1225',
                border: '1px solid #1f2937',
                borderRadius: '8px',
                fontSize: '12px',
                fontFamily: 'JetBrains Mono',
              }}
              labelStyle={{ color: '#9ca3af' }}
            />
            <Line
              type="monotone"
              dataKey="loss"
              stroke="#00d4ff"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
