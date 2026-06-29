import { useState } from 'react'
import { useClusterStatus } from '../hooks/useClusterStatus'
import { useWorkers } from '../hooks/useWorkers'
import { api } from '../lib/api'
import WorkerCard from '../components/WorkerCard'
import LossCurve from '../components/LossCurve'
import GradientHeatmap from '../components/GradientHeatmap'
import ThroughputBadge from '../components/ThroughputBadge'
import DriveFileTree from '../components/DriveFileTree'

function Skeleton() {
  return (
    <div className="animate-pulse space-y-6">
      <div className="h-8 bg-gray-800 rounded w-72" />
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
        {[1, 2, 3].map(i => <div key={i} className="h-32 bg-gray-800 rounded-lg" />)}
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div className="h-48 bg-gray-800 rounded-lg" />
        <div className="h-48 bg-gray-800 rounded-lg" />
      </div>
    </div>
  )
}

export default function ClusterOverview() {
  const { status, loading: statusLoading, error: statusError, refetch } = useClusterStatus()
  const { workers, loading: workersLoading } = useWorkers()
  const [aggregating, setAggregating] = useState(false)
  const [pausing, setPausing] = useState(false)

  const active = workers.filter(w => w.status === 'active')
  const stale = workers.filter(w => w.status === 'stale')
  const totalStep = status?.global_step || 0
  const targetSteps = status?.total_target_steps || 0
  const phase = status?.phase || 'idle'
  const isPaused = phase === 'paused'

  const layerNorms = active
    .filter(w => w.loss != null && w.loss > 0)
    .map(w => Math.max(0, Math.min(1, 1 - w.loss / 5)))

  async function handleAggregate() {
    setAggregating(true)
    try {
      await api.aggregate(totalStep)
    } catch { /* ignore */ }
    setTimeout(() => setAggregating(false), 3000)
  }

  async function handlePauseResume() {
    setPausing(true)
    try {
      if (isPaused) {
        await api.resumeTraining()
      } else {
        await api.pauseTraining()
      }
      setTimeout(refetch, 500)
    } catch { /* ignore */ }
    setTimeout(() => setPausing(false), 2000)
  }

  if (statusLoading && workersLoading) return <Skeleton />

  if (statusError && !status) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <div className="text-warning text-3xl mb-4">⚠</div>
        <p className="font-mono text-gray-500 text-sm mb-2">Cannot connect to coordinator</p>
        <p className="font-mono text-gray-700 text-xs max-w-md">{statusError}</p>
        <button onClick={refetch} className="mt-4 bg-gray-800 text-gray-400 text-xs font-mono px-3 py-1.5 rounded hover:bg-gray-700">Retry</button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="font-mono text-lg text-white">
            STEP <span className="text-accent">{totalStep.toLocaleString()}</span> / {targetSteps.toLocaleString()}
          </h1>
          <div className="flex items-center gap-2 mt-1">
            <span className={`w-1.5 h-1.5 rounded-full ${phase === 'training' ? 'bg-success' : phase === 'aggregating' ? 'bg-accent' : isPaused ? 'bg-warning' : 'bg-gray-600'}`} />
            <span className="text-[10px] font-mono text-gray-700 uppercase tracking-wider">{phase}</span>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <ThroughputBadge tokensPerSecond={status?.tokens_per_second_total} />
          <button
            onClick={handleAggregate}
            disabled={aggregating || isPaused}
            className="bg-accent/10 text-accent border border-accent/30 text-xs font-mono px-3 py-1.5 rounded hover:bg-accent/20 transition-colors disabled:opacity-40"
          >
            {aggregating ? '...' : 'FORCE AGGREGATE'}
          </button>
          <button
            onClick={handlePauseResume}
            disabled={pausing}
            className={`text-xs font-mono px-3 py-1.5 rounded transition-colors disabled:opacity-40 ${
              isPaused
                ? 'bg-success/10 text-success border border-success/30 hover:bg-success/20'
                : 'bg-warning/10 text-warning border border-warning/30 hover:bg-warning/20'
            }`}
          >
            {pausing ? '...' : isPaused ? 'RESUME' : 'PAUSE'}
          </button>
        </div>
      </div>

      {workers.length > 0 ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
          {active.map(w => <WorkerCard key={w.worker_id} worker={w} />)}
          {stale.map(w => <WorkerCard key={w.worker_id} worker={w} />)}
        </div>
      ) : (
        <div className="border border-dashed border-gray-800 rounded-lg py-16 text-center">
          <p className="text-gray-700 font-mono text-sm">No workers registered</p>
          <p className="text-gray-800 font-mono text-xs mt-1">Go to Setup to add workers</p>
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <LossCurve data={status?.total_loss_history || []} />
        <GradientHeatmap layerNorms={layerNorms} />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <DriveFileTree refreshTrigger={totalStep} />
        <div className="bg-[#0d1225] border border-gray-800 rounded-lg p-4">
          <h3 className="text-xs font-mono text-gray-600 uppercase tracking-wider mb-3">Cluster Info</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-xs font-mono">
            <div>
              <span className="text-gray-700 text-[10px] block">Active Workers</span>
              <span className="text-success text-sm">{active.length}</span>
            </div>
            <div>
              <span className="text-gray-700 text-[10px] block">Stale Workers</span>
              <span className="text-warning text-sm">{stale.length}</span>
            </div>
            <div>
              <span className="text-gray-700 text-[10px] block">Master Weights</span>
              <span className="text-gray-300 text-sm">v{status?.master_weights_version || 0}</span>
            </div>
            <div>
              <span className="text-gray-700 text-[10px] block">Learning Rate</span>
              <span className="text-gray-300 text-sm">{status?.current_lr?.toExponential() || '3e-4'}</span>
            </div>
            <div>
              <span className="text-gray-700 text-[10px] block">Aggregating</span>
              <span className={status?.aggregation_in_progress ? 'text-accent' : 'text-gray-600'}>
                {status?.aggregation_in_progress ? 'Yes' : 'No'}
              </span>
            </div>
            <div>
              <span className="text-gray-700 text-[10px] block">Phase</span>
              <span className="text-accent text-sm uppercase">{phase}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
