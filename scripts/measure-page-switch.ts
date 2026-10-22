// Times page switches in a running debug build (`bun dev`), which exposes
// Chrome DevTools on port 4000. Open a project with several pages first, on
// any page but the last ones; the script steps through pages and reports how
// long each switch takes to show the full page and how much main-thread time
// it costs. Run it right after opening the project, so the first jumps land
// on pages nothing prepared yet. It only presses arrows and clicks rail
// items; it edits nothing.
//
//   bun scripts/measure-page-switch.ts [steps] [--profile]
//
// With --profile it also records a CPU profile of a few paused steps and
// prints the functions that took the most main-thread time.

import { chromium, type Page } from 'playwright'

const steps = Number(process.argv.slice(2).find((arg) => /^\d+$/.test(arg)) ?? 12)
const profile = process.argv.includes('--profile')
const browser = await chromium.connectOverCDP('http://127.0.0.1:4000')
const page = browser
  .contexts()
  .flatMap((context) => context.pages())
  .find((candidate) => candidate.url().startsWith('http://localhost:3000'))
if (!page) throw new Error('No koharu window found on port 4000; is `bun dev` running?')
const cdp = await page.context().newCDPSession(page)
await cdp.send('Performance.enable')

const activeLabel = (target: Page) =>
  target.evaluate(
    () => document.querySelector('article[data-active=true] span.truncate')?.textContent ?? null,
  )

async function mainThreadMs(): Promise<number> {
  const { metrics } = await cdp.send('Performance.getMetrics')
  return (metrics.find((metric) => metric.name === 'TaskDuration')?.value ?? 0) * 1000
}

/** Runs `action`, then waits until the active page changed and its full frame shows. */
async function timeSwitch(action: () => Promise<void>, settle = 1500) {
  const before = await activeLabel(page!)
  const busyBefore = await mainThreadMs()
  const started = performance.now()
  await action()
  await page!.waitForFunction(
    (previous) =>
      document.querySelector('article[data-active=true] span.truncate')?.textContent !== previous &&
      !document.querySelector('[data-testid=page-preview]'),
    before,
    { polling: 'raf', timeout: 10_000 },
  )
  const shown = performance.now() - started
  await page!.waitForTimeout(settle)
  return { shown, busy: (await mainThreadMs()) - busyBefore }
}

function summary(label: string, values: number[]) {
  const sorted = [...values].sort((left, right) => left - right)
  const median = sorted[Math.floor(sorted.length / 2)] ?? NaN
  const max = sorted.at(-1) ?? NaN
  console.log(
    `${label}: median ${median.toFixed(0)} ms, max ${max.toFixed(0)} ms (${values.length})`,
  )
}

await page.keyboard.press('Escape')

// Jumps to the farthest rendered rail item, before any stepping, so
// no prefetch prepared it.
const jumps = []
for (let jump = 0; jump < 2; jump++) {
  const items = page.locator('article[data-active]')
  const count = await items.count()
  const activeIndex = await items.evaluateAll((articles) =>
    articles.findIndex((article) => article.getAttribute('data-active') === 'true'),
  )
  const target = activeIndex < count / 2 ? count - 1 : 0
  jumps.push(await timeSwitch(() => items.nth(target).click({ position: { x: 8, y: 8 } })))
}
summary(
  'jump across the rail, page shown',
  jumps.map((sample) => sample.shown),
)
summary(
  'jump across the rail, main thread',
  jumps.map((sample) => sample.busy),
)

const paused = []
for (let step = 0; step < steps; step++) {
  paused.push(await timeSwitch(() => page.keyboard.press('ArrowRight')))
}
summary(
  'step with a pause, page shown',
  paused.map((sample) => sample.shown),
)
summary(
  'step with a pause, main thread incl. prefetch',
  paused.map((sample) => sample.busy),
)

const quick = await timeSwitch(async () => {
  for (let step = 0; step < steps; step++) await page.keyboard.press('ArrowLeft')
})
console.log(
  `${steps} quick steps back: last page shown after ${quick.shown.toFixed(0)} ms, main thread ${quick.busy.toFixed(0)} ms`,
)

if (profile) {
  await cdp.send('Profiler.enable')
  await cdp.send('Profiler.setSamplingInterval', { interval: 200 })
  await cdp.send('Profiler.start')
  for (let step = 0; step < 6; step++) {
    await timeSwitch(() => page.keyboard.press(step % 2 ? 'ArrowLeft' : 'ArrowRight'))
  }
  const { profile: recorded } = await cdp.send('Profiler.stop')
  const self = new Map<string, number>()
  const ids = new Map(recorded.nodes.map((node) => [node.id, node]))
  const deltas = recorded.timeDeltas ?? []
  ;(recorded.samples ?? []).forEach((id, index) => {
    const frame = ids.get(id)?.callFrame
    if (!frame || frame.functionName === '(idle)' || frame.functionName === '(program)') return
    const file = frame.url.split('/').slice(-2).join('/')
    const key = `${frame.functionName || '(anonymous)'} ${file}:${frame.lineNumber + 1}`
    self.set(key, (self.get(key) ?? 0) + (deltas[index + 1] ?? 0) / 1000)
  })
  const total = [...self.values()].reduce((sum, value) => sum + value, 0)
  console.log(`profile of 6 paused steps: ${total.toFixed(0)} ms busy; top self time:`)
  for (const [key, ms] of [...self].sort((left, right) => right[1] - left[1]).slice(0, 15)) {
    console.log(`  ${ms.toFixed(1).padStart(7)} ms  ${key}`)
  }
  // Inclusive time of the app's own functions, which names the components
  // whose renders create the elements above.
  const parents = new Map<number, number>()
  for (const node of recorded.nodes) {
    for (const child of node.children ?? []) parents.set(child, node.id)
  }
  const inclusive = new Map<string, number>()
  ;(recorded.samples ?? []).forEach((id, index) => {
    const delta = (deltas[index + 1] ?? 0) / 1000
    const seen = new Set<string>()
    for (let node: number | undefined = id; node !== undefined; node = parents.get(node)) {
      const frame = ids.get(node)?.callFrame
      if (!frame?.url.includes('packages_koharu') || !frame.functionName) continue
      const key = `${frame.functionName} ${frame.url.split('/').slice(-1)[0]}:${frame.lineNumber + 1}`
      if (seen.has(key)) continue
      seen.add(key)
      inclusive.set(key, (inclusive.get(key) ?? 0) + delta)
    }
  })
  console.log('app functions by inclusive time:')
  for (const [key, ms] of [...inclusive].sort((left, right) => right[1] - left[1]).slice(0, 40)) {
    console.log(`  ${ms.toFixed(1).padStart(7)} ms  ${key}`)
  }
}

// Leave the app window open; only the DevTools connection ends.
process.exit(0)
