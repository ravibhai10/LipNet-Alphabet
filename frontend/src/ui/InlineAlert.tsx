import React from 'react'
import { Icon } from '../workspace/Icon'

interface InlineAlertProps {
  variant?: 'info' | 'warning' | 'error' | 'success'
  title?: string
  children: React.ReactNode
  actionLabel?: string
  onAction?: () => void
  onDismiss?: () => void
  className?: string
  style?: React.CSSProperties
}

export function InlineAlert({
  variant = 'info',
  title,
  children,
  actionLabel,
  onAction,
  onDismiss,
  className = '',
  style = {}
}: InlineAlertProps) {
  const getTheme = () => {
    switch (variant) {
      case 'error':
        return {
          bg: 'rgba(239, 68, 68, 0.08)',
          border: 'rgba(239, 68, 68, 0.22)',
          color: '#ef4444',
          icon: 'alert'
        }
      case 'warning':
        return {
          bg: 'rgba(245, 158, 11, 0.08)',
          border: 'rgba(245, 158, 11, 0.22)',
          color: '#f59e0b',
          icon: 'info'
        }
      case 'success':
        return {
          bg: 'rgba(16, 185, 129, 0.08)',
          border: 'rgba(16, 185, 129, 0.22)',
          color: '#10b981',
          icon: 'shield'
        }
      case 'info':
      default:
        return {
          bg: 'rgba(99, 102, 241, 0.08)',
          border: 'rgba(99, 102, 241, 0.22)',
          color: 'var(--ws-accent)',
          icon: 'info'
        }
    }
  }

  const theme = getTheme()

  return (
    <div
      role="alert"
      className={`saas-inline-alert ${className}`}
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '12px',
        padding: '12px 16px',
        borderRadius: '8px',
        background: theme.bg,
        border: `1px solid ${theme.border}`,
        fontSize: '13px',
        lineHeight: 1.5,
        margin: '12px 0',
        ...style
      }}
    >
      <div style={{ color: theme.color, marginTop: '2px', flexShrink: 0 }}>
        <Icon name={theme.icon} size={16} />
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        {title && (
          <div style={{ fontWeight: 600, color: 'var(--ws-text)', marginBottom: '2px' }}>
            {title}
          </div>
        )}
        <div style={{ color: 'var(--ws-text-muted)' }}>
          {children}
        </div>
      </div>

      {actionLabel && onAction && (
        <button
          type="button"
          onClick={onAction}
          className="ws-btn ws-btn-ghost ws-btn-sm"
          style={{
            alignSelf: 'center',
            fontSize: '11px',
            padding: '4px 10px',
            minHeight: '28px',
            flexShrink: 0
          }}
        >
          {actionLabel}
        </button>
      )}

      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss alert"
          style={{
            background: 'transparent',
            border: 'none',
            color: 'var(--ws-text-muted)',
            cursor: 'pointer',
            padding: '4px',
            fontSize: '14px',
            lineHeight: 1,
            alignSelf: 'flex-start'
          }}
        >
          ✕
        </button>
      )}
    </div>
  )
}
