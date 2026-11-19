import React from 'react'
import { Icon } from '../workspace/Icon'

interface EmptyStateProps {
  title: string
  description?: string
  icon?: string
  actionLabel?: string
  onAction?: () => void
  secondaryActionLabel?: string
  onSecondaryAction?: () => void
  children?: React.ReactNode
  className?: string
  style?: React.CSSProperties
}

export function EmptyState({
  title,
  description,
  icon = 'layers',
  actionLabel,
  onAction,
  secondaryActionLabel,
  onSecondaryAction,
  children,
  className = '',
  style = {}
}: EmptyStateProps) {
  return (
    <div
      className={`saas-empty-state ${className}`}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        padding: '48px 24px',
        border: '1px dashed var(--ws-border)',
        borderRadius: '12px',
        background: 'rgba(255, 255, 255, 0.01)',
        margin: '16px 0',
        ...style
      }}
    >
      <div
        style={{
          width: '44px',
          height: '44px',
          borderRadius: '10px',
          background: 'rgba(255, 255, 255, 0.04)',
          border: '1px solid var(--ws-border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--ws-text-muted)',
          marginBottom: '16px'
        }}
      >
        <Icon name={icon} size={20} />
      </div>

      <h3
        style={{
          fontSize: '15px',
          fontWeight: 600,
          color: 'var(--ws-text)',
          letterSpacing: '-0.015em',
          margin: '0 0 6px 0'
        }}
      >
        {title}
      </h3>

      {description && (
        <p
          style={{
            fontSize: '13px',
            color: 'var(--ws-text-muted)',
            maxWidth: '440px',
            lineHeight: 1.6,
            margin: '0 0 20px 0'
          }}
        >
          {description}
        </p>
      )}

      {children}

      {(actionLabel || secondaryActionLabel) && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: children ? '16px' : '0' }}>
          {actionLabel && onAction && (
            <button
              type="button"
              className="ws-btn ws-btn-primary"
              onClick={onAction}
              style={{
                minHeight: '36px',
                padding: '0 16px',
                fontSize: '12px',
                fontWeight: 600
              }}
            >
              {actionLabel}
            </button>
          )}

          {secondaryActionLabel && onSecondaryAction && (
            <button
              type="button"
              className="ws-btn ws-btn-ghost"
              onClick={onSecondaryAction}
              style={{
                minHeight: '36px',
                padding: '0 14px',
                fontSize: '12px'
              }}
            >
              {secondaryActionLabel}
            </button>
          )}
        </div>
      )}
    </div>
  )
}
