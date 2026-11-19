import React, { createContext, useContext, useState, useCallback, useMemo } from 'react'
import { Icon } from '../workspace/Icon'

export type ToastType = 'success' | 'error' | 'info' | 'warning'

export interface ToastItem {
  id: string
  title?: string
  message: string
  type?: ToastType
  duration?: number
}

interface ToastContextValue {
  toast: (item: Omit<ToastItem, 'id'> | string) => void
  success: (message: string, title?: string) => void
  error: (message: string, title?: string) => void
  info: (message: string, title?: string) => void
  warning: (message: string, title?: string) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) {
    // Graceful fallback if used outside provider
    return {
      toast: () => {},
      success: () => {},
      error: () => {},
      info: () => {},
      warning: () => {}
    }
  }
  return ctx
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([])

  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const toast = useCallback((item: Omit<ToastItem, 'id'> | string) => {
    const id = Math.random().toString(36).substring(2, 9)
    const toastObj: ToastItem = typeof item === 'string'
      ? { id, message: item, type: 'info' }
      : { id, type: 'info', ...item }

    setToasts(prev => [...prev.slice(-4), toastObj]) // Keep max 5

    const duration = toastObj.duration ?? 3500
    if (duration > 0) {
      setTimeout(() => {
        removeToast(id)
      }, duration)
    }
  }, [removeToast])

  const success = useCallback((message: string, title?: string) => toast({ message, title, type: 'success' }), [toast])
  const error = useCallback((message: string, title?: string) => toast({ message, title, type: 'error' }), [toast])
  const info = useCallback((message: string, title?: string) => toast({ message, title, type: 'info' }), [toast])
  const warning = useCallback((message: string, title?: string) => toast({ message, title, type: 'warning' }), [toast])

  const contextValue = useMemo(() => ({
    toast,
    success,
    error,
    info,
    warning
  }), [toast, success, error, info, warning])

  return (
    <ToastContext.Provider value={contextValue}>
      {children}
      {/* Toast floating container */}
      <div
        className="saas-toast-container"
        aria-live="polite"
        style={{
          position: 'fixed',
          bottom: '24px',
          right: '24px',
          zIndex: 99999,
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          maxWidth: '380px',
          width: 'calc(100vw - 48px)',
          pointerEvents: 'none'
        }}
      >
        {toasts.map(t => {
          const isError = t.type === 'error'
          const isSuccess = t.type === 'success'
          const isWarning = t.type === 'warning'

          const accentColor = isError ? '#ef4444' : isSuccess ? '#10b981' : isWarning ? '#f59e0b' : 'var(--ws-accent)'
          const iconName = isError ? 'alert' : isSuccess ? 'shield' : isWarning ? 'info' : 'info'

          return (
            <div
              key={t.id}
              className="saas-toast-item"
              style={{
                pointerEvents: 'auto',
                background: 'var(--ws-surface2)',
                border: '1px solid var(--ws-border-hi)',
                boxShadow: '0 8px 24px rgba(0, 0, 0, 0.35), 0 2px 6px rgba(0, 0, 0, 0.15)',
                borderRadius: '10px',
                padding: '12px 14px',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '10px',
                fontSize: '13px',
                animation: 'saas-toast-slide-in 0.22s cubic-bezier(0.16, 1, 0.3, 1)',
                color: 'var(--ws-text)'
              }}
            >
              <span style={{ color: accentColor, marginTop: '1px', flexShrink: 0 }}>
                <Icon name={iconName} size={16} />
              </span>

              <div style={{ flex: 1, minWidth: 0 }}>
                {t.title && (
                  <div style={{ fontWeight: 600, fontSize: '12px', color: 'var(--ws-text)', marginBottom: '2px' }}>
                    {t.title}
                  </div>
                )}
                <div style={{ color: 'var(--ws-text-muted)', fontSize: '12px', lineHeight: 1.4, wordBreak: 'break-word' }}>
                  {t.message}
                </div>
              </div>

              <button
                type="button"
                onClick={() => removeToast(t.id)}
                aria-label="Close notification"
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--ws-text-dim)',
                  cursor: 'pointer',
                  padding: '2px 4px',
                  fontSize: '14px',
                  lineHeight: 1
                }}
              >
                ✕
              </button>
            </div>
          )
        })}
      </div>
    </ToastContext.Provider>
  )
}
