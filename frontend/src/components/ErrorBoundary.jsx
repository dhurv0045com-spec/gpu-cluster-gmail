import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center py-24 text-center px-4">
          <div className="text-warning text-3xl mb-4">⚠</div>
          <h2 className="font-mono text-white text-lg mb-2">Something went wrong</h2>
          <p className="font-mono text-gray-500 text-sm mb-4 max-w-md">
            {this.state.error?.message || 'An unexpected error occurred'}
          </p>
          <button
            onClick={() => {
              this.setState({ hasError: false, error: null })
              window.location.reload()
            }}
            className="bg-accent text-deep text-sm font-mono px-4 py-2 rounded hover:bg-accent/90 transition-colors"
          >
            Reload
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
