import { memo, useMemo } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import '../styles/math.css'

interface Props { tex: string; inline?: boolean; className?: string }

/** Local font assets, genuine mathematical layout, and accessible MathML. */
export const MathExpression = memo(function MathExpression({ tex, inline = false, className = '' }: Props) {
  const result = useMemo(() => {
    try {
      return { html: katex.renderToString(tex, {
        displayMode: !inline, output: 'htmlAndMathml', throwOnError: true,
        strict: 'error', trust: false, maxExpand: 1000, maxSize: 10,
      }), error: false }
    } catch {
      return { html: '', error: true }
    }
  }, [tex, inline])
  if (result.error) return <span className="math-expression-error" role="status" data-math-status="error">This equation could not be displayed.</span>
  return <span className={`math-expression ${inline ? 'math-expression-inline' : 'math-expression-display'} ${className}`.trim()}
    data-math-status="ready" data-math-tex={tex} dangerouslySetInnerHTML={{ __html: result.html }}/>
})
