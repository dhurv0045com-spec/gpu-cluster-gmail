import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'

const steps = [
  { id: 0, label: 'Connect Drive' },
  { id: 1, label: 'Configure Cluster' },
  { id: 2, label: 'Initialize' },
  { id: 3, label: 'Launch Workers' },
]

export default function SetupWizard() {
  const [step, setStep] = useState(0)
  const [authenticated, setAuthenticated] = useState(false)
  const [checkingAuth, setCheckingAuth] = useState(true)
  const [authUrl, setAuthUrl] = useState('')
  const oauthWindow = useRef(null)
  const [folderId, setFolderId] = useState('')
  const [numWorkers, setNumWorkers] = useState(3)
  const [targetSteps, setTargetSteps] = useState(100000)
  const [checkpointFile, setCheckpointFile] = useState('anra_frontier_500m.pt')
  const [clusterId, setClusterId] = useState(null)
  const [workerLinks, setWorkerLinks] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let mounted = true
    let timer

    async function checkAuthentication() {
      try {
        const result = await api.getAuthStatus()
        if (!mounted) return
        setAuthenticated(result.authenticated)
        setCheckingAuth(false)
        if (result.authenticated) {
          setStep(current => Math.max(current, 1))
          oauthWindow.current?.close()
        }
      } catch (err) {
        if (mounted) {
          setCheckingAuth(false)
          setError(err.message)
        }
      }
    }

    checkAuthentication()
    timer = setInterval(checkAuthentication, 2000)
    return () => {
      mounted = false
      clearInterval(timer)
    }
  }, [])

  async function handleConnectDrive() {
    setLoading(true)
    setError(null)
    setAuthUrl('')
    // Opening synchronously preserves popup permission while the URL request runs.
    const popup = window.open('about:blank', 'anra-drive-oauth')
    oauthWindow.current = popup
    try {
      const result = await api.getAuthLogin()
      setAuthUrl(result.authorization_url)
      if (popup) popup.location.href = result.authorization_url
    } catch (err) {
      popup?.close()
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function validateStep1() {
    if (!folderId.trim()) return 'Folder ID is required'
    if (numWorkers < 1 || numWorkers > 20) return 'Workers must be 1-20'
    if (targetSteps < 1) return 'Target steps must be > 0'
    if (!checkpointFile.trim()) return 'Checkpoint filename is required'
    return null
  }

  async function handleInit() {
    if (!authenticated) {
      setStep(0)
      setError('Connect Google Drive before configuring the cluster')
      return
    }
    const validationError = validateStep1()
    if (validationError) {
      setError(validationError)
      return
    }
    setStep(2)
  }

  async function handleInitCluster() {
    setLoading(true)
    setError(null)
    try {
      const result = await api.initCluster({
        coordinator_drive_folder_id: folderId.trim(),
        master_checkpoint_filename: checkpointFile.trim(),
        total_target_steps: targetSteps,
      })
      setClusterId(result.cluster_id)

      const links = Array.from({ length: numWorkers }, (_, i) => {
        const wid = `worker_${String.fromCharCode(65 + i)}`
        return {
          workerId: wid,
          email: `account_${i + 1}@gmail.com`,
          notebookUrl: 'https://colab.research.google.com/github/dhurv0045com-spec/gpu-cluster-gmail/blob/main/worker/AN_RA_CLUSTER_WORKER.ipynb',
          config: `WORKER_ID="${wid}"\nACCOUNT_EMAIL="replace_with_worker_${i + 1}_gmail@gmail.com"`,
        }
      })
      setWorkerLinks(links)
      setStep(3)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const inputClass = 'w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm font-mono text-gray-200 focus:border-accent outline-none placeholder-gray-700'

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="font-mono text-lg text-white mb-6">Cluster Setup</h1>

      <div className="flex mb-8">
        {steps.map((s, i) => {
          const active = step >= s.id
          return (
            <div key={s.id} className="flex items-center flex-1">
              <div className={`flex items-center gap-2 transition-colors ${active ? 'text-accent' : 'text-gray-700'}`}>
                <div className={`w-7 h-7 rounded-full border flex items-center justify-center text-xs font-mono transition-all ${
                  active ? 'border-accent bg-accent/10 text-accent' : 'border-gray-700 text-gray-700'
                }`}>
                  {s.id}
                </div>
                <span className="text-xs font-mono hidden sm:block">{s.label}</span>
              </div>
              {i < steps.length - 1 && (
                <div className={`flex-1 h-px mx-3 transition-colors ${active ? 'bg-accent/30' : 'bg-gray-800'}`} />
              )}
            </div>
          )
        })}
      </div>

      {step === 0 && (
        <div className="bg-[#0d1225] border border-gray-800 rounded-lg p-6 space-y-5">
          <div>
            <h2 className="text-sm font-mono text-white mb-2">Connect the coordinator's Google Drive</h2>
            <p className="text-sm text-gray-500 font-mono">
              An-Ra uses one coordinator identity to publish state, sparse deltas, and recovery checkpoints.
              Worker accounts only need Editor access to the shared folder.
            </p>
          </div>

          <div className="bg-gray-900 rounded p-3 text-xs font-mono text-gray-500">
            Google will open in a new tab. Approve Drive access, then return here; this page checks the
            connection automatically every two seconds.
          </div>

          {error && (
            <div className="bg-warning/10 border border-warning/30 rounded px-3 py-2 text-warning text-xs font-mono">
              {error}
            </div>
          )}

          <button
            onClick={handleConnectDrive}
            disabled={loading || checkingAuth}
            className="bg-accent text-deep text-sm font-mono px-5 py-2 rounded hover:bg-accent/90 transition-colors disabled:opacity-40"
          >
            {checkingAuth ? 'Checking connection...' : loading ? 'Opening Google...' : 'Connect Google Drive'}
          </button>

          {authUrl && (
            <a href={authUrl} target="_blank" rel="noreferrer" className="block text-xs font-mono text-accent hover:underline">
              OAuth tab did not open? Continue here ↗
            </a>
          )}
        </div>
      )}

      {step === 1 && (
        <div className="bg-[#0d1225] border border-gray-800 rounded-lg p-6 space-y-5">
          <p className="text-sm text-gray-500 font-mono">
            Configure your cluster parameters. You'll need a Drive folder shared with all worker accounts.
          </p>

          <div>
            <label className="text-xs font-mono text-gray-600 block mb-1.5">
              Shared Drive Folder ID <span className="text-warning">*</span>
            </label>
            <input
              type="text"
              value={folderId}
              onChange={e => setFolderId(e.target.value)}
              placeholder="e.g. 1ABCxyz..."
              className={inputClass}
            />
            <p className="text-[10px] text-gray-700 mt-1 font-mono">
              Found in the URL when you open the Drive folder: drive.google.com/drive/folders/<span className="text-accent/60">FOLDER_ID</span>
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-mono text-gray-600 block mb-1.5">Number of Workers</label>
              <input
                type="number"
                value={numWorkers}
                onChange={e => setNumWorkers(Math.max(1, Math.min(20, parseInt(e.target.value) || 1)))}
                min={1} max={20}
                className={inputClass}
              />
            </div>
            <div>
              <label className="text-xs font-mono text-gray-600 block mb-1.5">Total Target Steps</label>
              <input
                type="number"
                value={targetSteps}
                onChange={e => setTargetSteps(Math.max(1, parseInt(e.target.value) || 0))}
                className={inputClass}
              />
            </div>
          </div>

          <div>
            <label className="text-xs font-mono text-gray-600 block mb-1.5">Checkpoint Filename</label>
            <input
              type="text"
              value={checkpointFile}
              onChange={e => setCheckpointFile(e.target.value)}
              className={inputClass}
            />
          </div>

          {error && (
            <div className="bg-warning/10 border border-warning/30 rounded px-3 py-2 text-warning text-xs font-mono">
              {error}
            </div>
          )}

          <button
            onClick={handleInit}
            className="bg-accent text-deep text-sm font-mono px-5 py-2 rounded hover:bg-accent/90 transition-colors"
          >
            Next →
          </button>
        </div>
      )}

      {step === 2 && (
        <div className="bg-[#0d1225] border border-gray-800 rounded-lg p-6 space-y-4">
          <p className="text-sm text-gray-500 font-mono">
            Initialize the cluster and create the coordinator state on Drive.
          </p>

          <div className="bg-gray-900 rounded p-3 text-xs font-mono">
            <pre className="text-gray-500">{JSON.stringify({
              folder_id: folderId,
              checkpoint: checkpointFile,
              target_steps: targetSteps,
              workers: numWorkers,
            }, null, 2)}</pre>
          </div>

          {error && (
            <div className="bg-warning/10 border border-warning/30 rounded px-3 py-2 text-warning text-xs font-mono">
              {error}
            </div>
          )}

          <div className="flex gap-3">
            <button
              onClick={() => setStep(1)}
              className="bg-gray-800 text-gray-400 text-sm font-mono px-4 py-2 rounded hover:bg-gray-700 transition-colors"
            >
              ← Back
            </button>
            <button
              onClick={handleInitCluster}
              disabled={loading}
              className="bg-accent text-deep text-sm font-mono px-5 py-2 rounded hover:bg-accent/90 transition-colors disabled:opacity-40 flex items-center gap-2"
            >
              {loading ? (
                <>
                  <span className="inline-block w-3 h-3 border-2 border-deep border-t-transparent rounded-full animate-spin" />
                  Initializing...
                </>
              ) : (
                'Initialize Cluster'
              )}
            </button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="bg-[#0d1225] border border-gray-800 rounded-lg p-6 space-y-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-success text-lg">✓</span>
            <p className="text-sm text-gray-400 font-mono">
              Cluster <span className="text-accent">{clusterId}</span> initialized.
              Open each notebook link in a Colab tab signed into the corresponding account.
            </p>
          </div>

          {workerLinks.map((wl, i) => (
            <div key={wl.workerId} className="border border-gray-800 rounded-lg p-4 space-y-3 hover:border-gray-700 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full bg-success`} />
                  <span className="text-sm font-mono text-accent">{wl.workerId}</span>
                  <span className="text-[10px] font-mono text-gray-700 bg-gray-900 px-1.5 py-0.5 rounded">Slot {i + 1}</span>
                </div>
                <span className="text-xs font-mono text-gray-700">{wl.email}</span>
              </div>
              <div className="bg-gray-900 rounded p-2.5 text-xs font-mono text-gray-400 whitespace-pre overflow-x-auto">
                {wl.config}
              </div>
              <a
                href={wl.notebookUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 bg-gray-800 text-accent text-xs font-mono px-3 py-1.5 rounded hover:bg-gray-700 transition-colors"
              >
                Open in Colab ↗
              </a>
            </div>
          ))}

          <a
            href="/cluster"
            className="inline-block mt-2 bg-accent/10 text-accent border border-accent/30 text-sm font-mono px-4 py-2 rounded hover:bg-accent/20 transition-colors"
          >
            Go to Dashboard →
          </a>
        </div>
      )}
    </div>
  )
}
