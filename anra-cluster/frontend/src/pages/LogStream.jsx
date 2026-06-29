import { useRef, useEffect, useState } from 'react'
import { useLogStream } from '../hooks/useLogStream'

const LEVEL_COLORS = {
  ERROR: 'text-red-400',
  WARN: 'text-warning',
  INFO: 'text-gray-400',
  DEBUG: 'text-gray-700',
}

function parseLevel(msg) {
  const match = msg.match(/^\[(\w+)\]/)
  return match ? match[1] : 'INFO'
}

export default function LogStream() {
  const { logs, connected } = useLogStream(500)
  const containerRef = useRef(null)
  const autoScrollRef = useRef(true)
  const [filter, setFilter] = useState('ALL')

  useEffect(() => {
    if (autoScrollRef.current && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [logs])

  function handleScroll() {
    if (!containerRef.current) return
    const el = containerRef.current
    autoScrollRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 50
  }

  const filteredLogs = filter === 'ALL' ? logs : logs.filter(msg => {
    const level = parseLevel(msg)
    return level === filter
  })

  const logCounts = logs.reduce((acc, msg) => {
    const level = parseLevel(msg)
    acc[level] = (acc[level] || 0) + 1
    return acc
  }, {})

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <h1 className="font-mono text-lg text-white">Live Log Stream</h1>
        <div className="flex items-center gap-3">
          <div className="flex gap-1">
            {['ALL', 'ERROR', 'WARN', 'INFO'].map(level => (
              <button
                key={level}
                onClick={() => setFilter(level)}
                className={`text-[10px] font-mono px-2 py-1 rounded transition-colors ${
                  filter === level
                    ? 'bg-accent/20 text-accent border border-accent/30'
                    : 'bg-gray-900 text-gray-600 border border-gray-800 hover:text-gray-400'
                }`}
              >
                {level}
                {level !== 'ALL' && logCounts[level] ? ` (${logCounts[level]})` : ''}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-success shadow-[0_0_6px_rgba(0,255,136,0.5)]' : 'bg-warning'}`} />
            <span className="text-xs font-mono text-gray-600">{connected ? 'Connected' : 'Reconnecting...'}</span>
          </div>
        </div>
      </div>

      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="bg-[#0d1225] border border-gray-800 rounded-lg p-4 h-[65vh] overflow-y-auto font-mono text-xs"
      >
        {filteredLogs.length === 0 ? (
          <div className="text-gray-800 py-12 text-center">
            {logs.length === 0 ? 'Waiting for log messages...' : 'No matching log entries'}
          </div>
        ) : (
          filteredLogs.map((msg, i) => {
            const level = parseLevel(msg)
            return (
              <div
                key={i}
                className={`py-0.5 leading-5 transition-colors hover:opacity-80 ${LEVEL_COLORS[level] || LEVEL_COLORS.INFO}`}
              >
                {msg}
              </div>
            )
          })
        )}
      </div>

      <div className="flex gap-2">
        <button
          onClick={() => {
            autoScrollRef.current = true
            if (containerRef.current) containerRef.current.scrollTop = containerRef.current.scrollHeight
          }}
          className="bg-gray-800 text-gray-500 text-xs font-mono px-3 py-1.5 rounded hover:bg-gray-700 transition-colors"
        >
          ↓ Auto-scroll
        </button>
        <button
          onClick={() => {
            if (containerRef.current) containerRef.current.scrollTop = containerRef.current.scrollHeight
          }}
          className="bg-gray-800 text-gray-500 text-xs font-mono px-3 py-1.5 rounded hover:bg-gray-700 transition-colors"
        >
          Bottom
        </button>
        <span className="text-[10px] font-mono text-gray-700 self-center ml-auto">
          {logs.length} entries
          {filter !== 'ALL' && ` (${filteredLogs.length} shown)`}
        </span>
      </div>
    </div>
  )
}
