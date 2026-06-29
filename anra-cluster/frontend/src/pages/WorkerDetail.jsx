import { useParams, useNavigate } from 'react-router-dom'
import { useWorkers } from '../hooks/useWorkers'
import { useClusterStatus } from '../hooks/useClusterStatus'
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid } from 'recharts'

function MetricCard({ label, value }) {
  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-lg p-3">
      <div className="text-[10px] font-mono text-gray-700 mb-1 uppercase tracking-wider">{label}</div>
      <div className="text-sm font-mono text-gray-200">{value}</div>
    </div>
  )
}

export default function WorkerDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { workers } = useWorkers(3000)
  const { status } = useClusterStatus(5000)

  if (!id) {
    return (
      <div className="space-y-4">
        <h1 className="font-mono text-lg text-white mb-4">All Workers</h1>
        {workers.length === 0 ? (
          <div className="border border-dashed border-gray-800 rounded-lg py-16 text-center">
            <p className="text-gray-700 font-mono text-sm">No workers registered</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {workers.map(w => (
              <div
                key={w.worker_id}
                onClick={() => navigate(`/workers/${w.worker_id}`)}
                className="bg-[#0d1225] border border-gray-800 rounded-lg p-4 cursor-pointer hover:border-accent/40 transition-all"
              >
                <div className="flex items-center gap-2 mb-3">
                  <span className={`w-2 h-2 rounded-full ${w.status === 'active' ? 'bg-success' : 'bg-warning'}`} />
                  <span className="font-mono text-sm text-gray-200">{w.worker_id}</span>
                  <span className="text-[10px] text-gray-700 font-mono bg-gray-900 px-1.5 py-0.5 rounded">S{w.assigned_slot}</span>
                </div>
                <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-xs font-mono text-gray-500">
                  <div>Step <span className="text-gray-200">{w.current_step}</span></div>
                  <div>Loss <span className="text-gray-200">{w.loss?.toFixed(4) || '—'}</span></div>
                  <div>Tokens <span className="text-gray-200">{(w.tokens_processed || 0).toLocaleString()}</span></div>
                  <div>GPU <span className="text-gray-200">{w.gpu_memory_mb?.toFixed(0) || '?'} MB</span></div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  const worker = workers.find(w => w.worker_id === id)

  if (!worker) {
    return (
      <div className="flex flex-col items-center justify-center py-24">
        <p className="font-mono text-gray-700">Worker not found</p>
        <button onClick={() => navigate('/workers')} className="text-accent text-sm font-mono mt-3 hover:underline">
          ← Back to all workers
        </button>
      </div>
    )
  }

  const lossHistory = (status?.total_loss_history || [])
    .map((v, i) => ({ step: i, loss: typeof v === 'number' ? v : 0 }))

  return (
    <div className="space-y-6 max-w-4xl">
      <button onClick={() => navigate('/workers')} className="text-accent text-xs font-mono hover:underline flex items-center gap-1">
        ← Back to workers
      </button>

      <div className="bg-[#0d1225] border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-3 mb-6">
          <span className={`w-3 h-3 rounded-full ${worker.status === 'active' ? 'bg-success shadow-[0_0_8px_rgba(0,255,136,0.3)]' : 'bg-warning'}`} />
          <h1 className="font-mono text-lg text-white">{worker.worker_id}</h1>
          <span className="text-xs font-mono text-gray-700 bg-gray-900 px-2 py-0.5 rounded">Slot {worker.assigned_slot}</span>
          <span className="text-[10px] font-mono text-gray-700">{worker.account_email}</span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          <MetricCard label="Current Step" value={worker.current_step?.toLocaleString() || '0'} />
          <MetricCard label="Loss" value={worker.loss?.toFixed(4) || '—'} />
          <MetricCard label="Tokens Processed" value={(worker.tokens_processed || 0).toLocaleString()} />
          <MetricCard label="GPU Memory" value={worker.gpu_memory_mb ? `${worker.gpu_memory_mb.toFixed(0)} MB` : '—'} />
        </div>

        <div className="h-64">
          <h3 className="text-xs font-mono text-gray-600 mb-3 uppercase tracking-wider">Loss History</h3>
          {lossHistory.length > 1 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lossHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1a2040" />
                <XAxis dataKey="step" tick={{ fill: '#4b5563', fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis domain={['dataMin - 0.1', 'dataMax + 0.1']} tick={{ fill: '#4b5563', fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0d1225', border: '1px solid #1f2937', borderRadius: '8px', fontSize: '12px', fontFamily: 'JetBrains Mono' }}
                  labelStyle={{ color: '#9ca3af' }}
                />
                <Line type="monotone" dataKey="loss" stroke="#00d4ff" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-700 font-mono text-xs">
              Insufficient data for loss curve
            </div>
          )}
        </div>
      </div>

      <div className="bg-[#0d1225] border border-gray-800 rounded-lg p-4">
        <h3 className="text-xs font-mono text-gray-600 uppercase tracking-wider mb-3">Raw Heartbeat Data</h3>
        <pre className="text-xs font-mono text-gray-500 bg-gray-900 rounded p-3 overflow-x-auto">
          {JSON.stringify(worker, null, 2)}
        </pre>
      </div>
    </div>
  )
}
