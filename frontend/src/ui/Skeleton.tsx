import React from 'react'

interface SkeletonProps {
  className?: string
  width?: string | number
  height?: string | number
  borderRadius?: string | number
  variant?: 'text' | 'rect' | 'circle' | 'card'
  style?: React.CSSProperties
}

export function Skeleton({
  className = '',
  width,
  height,
  borderRadius,
  variant = 'rect',
  style = {}
}: SkeletonProps) {
  const getVariantStyles = (): React.CSSProperties => {
    switch (variant) {
      case 'circle':
        return {
          borderRadius: '50%',
          width: width ?? '40px',
          height: height ?? '40px',
        }
      case 'text':
        return {
          borderRadius: '4px',
          height: height ?? '14px',
          width: width ?? '100%',
        }
      case 'card':
        return {
          borderRadius: '12px',
          height: height ?? '160px',
          width: width ?? '100%',
        }
      case 'rect':
      default:
        return {
          borderRadius: borderRadius ?? '8px',
          width: width ?? '100%',
          height: height ?? '24px',
        }
    }
  }

  return (
    <div
      className={`saas-skeleton ${className}`}
      aria-hidden="true"
      style={{
        background: 'linear-gradient(90deg, rgba(255, 255, 255, 0.04) 25%, rgba(255, 255, 255, 0.08) 50%, rgba(255, 255, 255, 0.04) 75%)',
        backgroundSize: '200% 100%',
        animation: 'saas-skeleton-shimmer 1.8s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        ...getVariantStyles(),
        ...style,
      }}
    />
  )
}

/** Pre-composed Skeleton card for loading stats or metrics */
export function MetricCardSkeleton() {
  return (
    <div
      style={{
        padding: '20px',
        borderRadius: '12px',
        border: '1px solid var(--ws-border)',
        background: 'var(--ws-surface)',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Skeleton variant="text" width="40%" height={12} />
        <Skeleton variant="circle" width={18} height={18} />
      </div>
      <Skeleton variant="text" width="60%" height={32} />
      <Skeleton variant="text" width="80%" height={10} />
    </div>
  )
}
