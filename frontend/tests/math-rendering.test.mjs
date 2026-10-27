import test from 'node:test'
import assert from 'node:assert/strict'
import katex from 'katex'
import { EQUATIONS } from '../src/lib/equations.ts'

for (const [name, tex] of Object.entries(EQUATIONS)) {
  test(`${name} renders accessible mathematical notation without parser errors`, () => {
    const html = katex.renderToString(tex, {
      displayMode: true, output: 'htmlAndMathml', throwOnError: true,
      strict: 'error', trust: false,
    })
    assert.match(html, /class="katex"/)
    assert.match(html, /<math /)
    assert.match(html, /application\/x-tex/)
    assert.doesNotMatch(html, /katex-error/)
    if (tex.includes('\\frac')) assert.match(html, /class="mfrac"/)
    if (tex.includes('\\sqrt')) assert.match(html, /<msqrt>/)
  })
}
