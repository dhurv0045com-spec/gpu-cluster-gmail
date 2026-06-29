import { useState, useEffect } from 'react'
import { api } from '../lib/api'

function formatSize(bytes) {
  if (!bytes) return '—'
  const mb = parseInt(bytes) / (1024 * 1024)
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${(parseInt(bytes) / 1024).toFixed(1)} KB`
}

export default function DriveFileTree({ refreshTrigger = 0 }) {
  const [files, setFiles] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    setLoading(true)
    api.getDriveFiles()
      .then(data => { if (mounted) setFiles(Array.isArray(data) ? data : []) })
      .catch(() => {})
      .finally(() => { if (mounted) setLoading(false) })
    return () => { mounted = false }
  }, [refreshTrigger])

  const totalSize = files.reduce((acc, f) => acc + (parseInt(f.size) || 0), 0)

  return (
    <div className="bg-[#0d1225] border border-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-mono text-gray-500 uppercase tracking-wider">Drive Files</h3>
        <span className="text-[10px] font-mono text-gray-600">
          {files.length} files / {formatSize(totalSize)}
        </span>
      </div>
      {loading ? (
        <div className="text-xs font-mono text-gray-600">Loading...</div>
      ) : files.length === 0 ? (
        <div className="text-xs font-mono text-gray-600">No files found</div>
      ) : (
        <div className="space-y-1 max-h-48 overflow-y-auto">
          {files.map(f => (
            <div key={f.id} className="flex items-center justify-between text-xs font-mono text-gray-400 py-0.5">
              <span className="truncate flex-1">
                {f.mimeType === 'application/vnd.google-apps.folder' ? '📁' : '📄'} {f.name}
              </span>
              <span className="text-gray-600 ml-2">{formatSize(f.size)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
