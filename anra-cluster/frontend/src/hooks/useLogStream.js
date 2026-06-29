import { useState, useEffect, useRef, useCallback } from 'react'
import { createLogStream } from '../lib/api'

export function useLogStream(maxLines = 500) {
  const [logs, setLogs] = useState([])
  const [connected, setConnected] = useState(false)
  const esRef = useRef(null)
  const retriesRef = useRef(0)

  const connect = useCallback(() => {
    esRef.current?.close()
    const es = createLogStream()
    esRef.current = es

    es.onopen = () => {
      setConnected(true)
      retriesRef.current = 0
    }

    es.onerror = () => {
      setConnected(false)
      es.close()
      retriesRef.current++
      const delay = Math.min(1000 * 2 ** retriesRef.current, 30000)
      setTimeout(connect, delay)
    }

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        setLogs(prev => {
          const next = [...prev, data.message]
          return next.length > maxLines ? next.slice(-maxLines) : next
        })
      } catch {
        // skip malformed
      }
    }
  }, [maxLines])

  useEffect(() => {
    connect()
    return () => esRef.current?.close()
  }, [connect])

  return { logs, connected }
}
