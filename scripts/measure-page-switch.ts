// Times page switches in a running debug build (`bun dev`), which exposes
// Chrome DevTools on port 4000. Open a project with several pages first, on
// any page but the last ones; the script steps through pages and reports how
// long each switch takes to show the full page and how much main-thread time
// it costs. Run it right after opening the project, so the first jumps land
// on pages nothing prepared yet. It only presses arrows and clicks rail
// items; it edits nothing.
//
//   bun scripts/measure-page-switch.ts [steps]

import { chromium, type Page } from 'playwright'

const steps = Number(process.argv[2] ?? 12)
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

// Leave the app window open; only the DevTools connection ends.
process.exit(0)
