import { useEffect, useMemo, useRef } from 'react'

const COLS = 64
const CELL_W = 4
const CELL_H = 4
const GAP = 1

export default function GradientHeatmap({ layerNorms = [] }) {
  const canvasRef = useRef(null)
  const animRef = useRef(null)
  const normalizedNorms = useMemo(() => {
    const logged = layerNorms.map(value => Math.log1p(Math.max(0, Number(value) || 0)))
    const maximum = Math.max(...logged, 0)
    return maximum > 0 ? logged.map(value => value / maximum) : logged
  }, [layerNorms])
  const normsRef = useRef(normalizedNorms)

  useEffect(() => {
    normsRef.current = normalizedNorms
  }, [normalizedNorms])

  const rows = Math.max(layerNorms.length, 1)
  const width = COLS * (CELL_W + GAP) - GAP
  const height = rows * (CELL_H + GAP) - GAP

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || rows === 0) return

    let phase = 0

    function draw() {
      const ctx = canvas.getContext('2d')
      const imageData = ctx.createImageData(width, height)
      const norms = normsRef.current
      phase += 0.02

      for (let r = 0; r < rows; r++) {
        const baseNorm = norms[r] || 0
        const pulse = 0.5 + 0.5 * Math.sin(phase + r * 0.3)
        const norm = baseNorm * pulse
        const intensity = Math.min(Math.floor(norm * 255 * 2), 255)
        const rVal = Math.min(Math.floor(40 * (norm * 5)), 100)
        const gVal = Math.min(Math.floor(212 * (norm * 5)), 255)
        const bVal = Math.min(Math.floor(255 * (norm * 3)), 255)

        for (let c = 0; c < COLS; c++) {
          const x = c * (CELL_W + GAP)
          const y = r * (CELL_H + GAP)
          const noise = 0.85 + 0.15 * Math.sin(phase * 2 + c * 0.5 + r * 0.7)
          const finalIntensity = Math.min(Math.floor(intensity * noise), 255)

          for (let py = 0; py < CELL_H; py++) {
            for (let px = 0; px < CELL_W; px++) {
              const idx = ((y + py) * width + (x + px)) * 4
              imageData.data[idx] = rVal
              imageData.data[idx + 1] = gVal
              imageData.data[idx + 2] = bVal
              imageData.data[idx + 3] = finalIntensity
            }
          }
        }
      }
      ctx.putImageData(imageData, 0, 0)
      animRef.current = requestAnimationFrame(draw)
    }

    animRef.current = requestAnimationFrame(draw)
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current)
    }
  }, [rows, width, height])

  if (layerNorms.length === 0) {
    return (
      <div className="bg-[#0d1225] border border-gray-800 rounded-lg p-4">
        <h3 className="text-xs font-mono text-gray-600 uppercase tracking-wider mb-3">Gradient Heatmap</h3>
        <div className="flex items-center justify-center h-[calc(1*(4+1)-1+32px)] text-gray-700 font-mono text-xs">
          Waiting for workers...
        </div>
      </div>
    )
  }

  return (
    <div className="bg-[#0d1225] border border-gray-800 rounded-lg p-4">
      <h3 className="text-xs font-mono text-gray-600 uppercase tracking-wider mb-3">Gradient Heatmap</h3>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="rounded"
        style={{ imageRendering: 'pixelated', width: `${width}px`, height: `${height}px` }}
      />
      <div className="flex justify-between text-[10px] font-mono text-gray-700 mt-2">
        <span>Layer 0</span>
        <span>{layerNorms.length} layers · log-relative L2 norm</span>
      </div>
    </div>
  )
}
