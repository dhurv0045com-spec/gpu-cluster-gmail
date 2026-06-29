import { useState, useEffect, useCallback, useRef } from 'react'
import { api } from '../lib/api'

export function useClusterStatus(pollInterval = 5000) {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const mountedRef = useRef(true)

  const fetch = useCallback(async () => {
    try {
      const data = await api.getTrainingStatus()
      if (mountedRef.current) {
        setStatus(data)
        setError(null)
      }
    } catch (err) {
      if (mountedRef.current) setError(err.message)
    } finally {
      if (mountedRef.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    fetch()
    const interval = setInterval(fetch, pollInterval)
    return () => {
      mountedRef.current = false
      clearInterval(interval)
    }
  }, [fetch, pollInterval])

  return { status, loading, error, refetch: fetch }
}
