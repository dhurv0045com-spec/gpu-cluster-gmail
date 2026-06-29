import { useState, useEffect, useCallback, useRef } from 'react'
import { api } from '../lib/api'

export function useWorkers(pollInterval = 5000) {
  const [workers, setWorkers] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const mountedRef = useRef(true)

  const fetch = useCallback(async () => {
    try {
      const data = await api.getWorkers()
      if (mountedRef.current) {
        setWorkers(Array.isArray(data) ? data : [])
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

  return { workers, loading, error, refetch: fetch }
}
