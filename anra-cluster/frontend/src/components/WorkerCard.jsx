import { useNavigate } from 'react-router-dom'

const statusColor = {
  active: 'bg-success shadow-[0_0_8px_rgba(0,255,136,0.3)]',
  stale: 'bg-warning shadow-[0_0_8px_rgba(255,107,53,0.3)]',
  inactive: 'bg-gray-600',
}

export default function WorkerCard({ worker }) {
  const navigate = useNavigate()
  const slot = worker.assigned_slot || '?'
  const memPct = Math.min((worker.gpu_memory_mb || 0) / 15100 * 100, 100)

  const memColor = memPct > 80 ? 'bg-warning' : memPct > 50 ? 'bg-accent' : 'bg-success'

  const timeSinceHb = worker.last_heartbeat
    ? `${Math.round((Date.now() / 1000 - worker.last_heartbeat) / 60)}m ago`
    : 'never'

  return (
    <div
      onClick={() => navigate(`/workers/${worker.worker_id}`)}
      className="bg-[#0d1225] border border-gray-800 rounded-lg p-4 cursor-pointer hover:border-accent/40 hover:shadow-[0_0_16px_rgba(0,212,255,0.06)] transition-all group"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rounded-full ${statusColor[worker.status] || statusColor.inactive}`} />
          <span className="font-mono text-sm text-gray-300 group-hover:text-accent transition-colors">{worker.worker_id}</span>
          <span className="text-[10px] text-gray-700 font-mono bg-gray-900 px-1.5 py-0.5 rounded">S{slot}</span>
        </div>
        <span className="font-mono text-[10px] text-gray-700">{timeSinceHb}</span>
      </div>

      <div className="grid grid-cols-3 gap-3 text-xs font-mono mb-3">
        <div>
          <span className="text-gray-700 block text-[10px]">Step</span>
          <span className="text-gray-200">{worker.current_step?.toLocaleString() || 0}</span>
        </div>
        <div>
          <span className="text-gray-700 block text-[10px]">Loss</span>
          <span className="text-gray-200">{worker.loss?.toFixed(4) || '—'}</span>
        </div>
        <div>
          <span className="text-gray-700 block text-[10px]">Tokens</span>
          <span className="text-gray-200">{(worker.tokens_processed || 0).toLocaleString()}</span>
        </div>
      </div>

      <div>
        <div className="flex justify-between text-[10px] font-mono mb-1">
          <span className="text-gray-700">GPU</span>
          <span className="text-gray-500">{worker.gpu_memory_mb?.toFixed(0) || '?'} / 15100 MB</span>
        </div>
        <div className="h-1.5 bg-gray-900 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${memColor}`}
            style={{ width: `${memPct}%` }}
          />
        </div>
      </div>
    </div>
  )
}
