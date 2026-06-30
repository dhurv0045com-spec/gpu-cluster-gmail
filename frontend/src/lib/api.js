const BASE = import.meta.env.VITE_API_URL || ''

async function request(path, options = {}) {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), options.timeout || 15000)

  try {
    const res = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      signal: controller.signal,
      ...options,
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      let detail = text || res.statusText
      try {
        const json = JSON.parse(text)
        detail = json.detail || detail
      } catch { /* response was not JSON */ }
      throw new Error(`API ${res.status}: ${detail}`)
    }
    return res.json()
  } finally {
    clearTimeout(timeout)
  }
}

export const api = {
  getAuthLogin: () => request('/api/auth/login'),

  getAuthStatus: () => request('/api/auth/status'),

  initCluster: (data) => request('/api/cluster/init', {
    method: 'POST', body: JSON.stringify(data), timeout: 30000,
  }),

  registerWorker: (data) => request('/api/workers/register', {
    method: 'POST', body: JSON.stringify(data),
  }),

  getWorkers: () => request('/api/workers'),

  heartbeat: (workerId, data) => request(`/api/workers/${workerId}/heartbeat`, {
    method: 'POST', body: JSON.stringify(data),
  }),

  gradientReady: (workerId, data) => request(`/api/workers/${workerId}/gradient_ready`, {
    method: 'POST', body: JSON.stringify(data),
  }),

  getTrainingStatus: () => request('/api/training/status'),

  aggregate: (step) => request('/api/training/aggregate', {
    method: 'POST', body: JSON.stringify({ step }),
  }),

  pauseTraining: () => request('/api/training/pause', { method: 'POST' }),

  resumeTraining: () => request('/api/training/resume', { method: 'POST' }),

  getDriveFiles: () => request('/api/drive/files'),

  getHealth: () => request('/api/health'),
}

export function createLogStream() {
  return new EventSource(`${BASE}/api/logs/stream`)
}
