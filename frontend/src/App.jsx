import { useState } from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import ErrorBoundary from './components/ErrorBoundary'
import ClusterOverview from './pages/ClusterOverview'
import SetupWizard from './pages/SetupWizard'
import WorkerDetail from './pages/WorkerDetail'
import LogStream from './pages/LogStream'

const navItems = [
  { path: '/cluster', label: 'Cluster', icon: '◈' },
  { path: '/setup', label: 'Setup', icon: '⚙' },
  { path: '/workers', label: 'Workers', icon: '⊞' },
  { path: '/logs', label: 'Logs', icon: '≡' },
]

export default function App() {
  const [mobileOpen, setMobileOpen] = useState(false)

  const nav = (
    <nav className={`${mobileOpen ? 'flex' : 'hidden'} md:flex flex-col md:flex-row gap-4 md:gap-6`}>
      {navItems.map(item => (
        <NavLink
          key={item.path}
          to={item.path}
          onClick={() => setMobileOpen(false)}
          className={({ isActive }) =>
            `flex items-center gap-1.5 text-sm font-mono transition-colors ${
              isActive ? 'text-accent' : 'text-gray-500 hover:text-gray-300'
            }`
          }
        >
          <span>{item.icon}</span>
          {item.label}
        </NavLink>
      ))}
    </nav>
  )

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-deep flex flex-col">
        <header className="border-b border-gray-800 px-4 md:px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-accent text-xl font-bold font-mono">AN-RA</span>
            <span className="text-gray-500 text-sm font-mono hidden sm:inline">CLUSTER</span>
          </div>

          {nav}

          <button
            className="md:hidden text-gray-400 hover:text-white text-xl"
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label="Toggle menu"
          >
            {mobileOpen ? '✕' : '☰'}
          </button>
        </header>

        {mobileOpen && (
          <div className="md:hidden border-b border-gray-800 px-4 py-3 bg-gray-900/50">
            {nav}
          </div>
        )}

        <main className="flex-1 p-4 md:p-6">
          <Routes>
            <Route path="/" element={<ClusterOverview />} />
            <Route path="/cluster" element={<ClusterOverview />} />
            <Route path="/setup" element={<SetupWizard />} />
            <Route path="/workers" element={<WorkerDetail />} />
            <Route path="/workers/:id" element={<WorkerDetail />} />
            <Route path="/logs" element={<LogStream />} />
          </Routes>
        </main>
      </div>
    </ErrorBoundary>
  )
}
