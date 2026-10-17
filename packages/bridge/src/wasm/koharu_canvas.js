// Browser 2D Canvas implementation of the koharu_canvas WASM interface for web preview environments.

export default async function initialize() {
  return undefined
}

export async function createCanvas(element) {
  const ctx = element.getContext('2d')
  let width = element.clientWidth || 800
  let height = element.clientHeight || 600
  let dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1
  let background = [255, 255, 255, 255]
  let zoom = 1
  let tx = 0
  let ty = 0
  let nextToken = 1
  const stagedFrames = new Map()
  const cachedPages = new Map()
  let activeFrame = null
  const opacityPreviews = new Map()
  let transformPreview = new Map()
  let activeStroke = null

  function parseManifest(bytes) {
    try {
      const text = new TextDecoder().decode(bytes)
      if (text.startsWith('{')) {
        return JSON.parse(text)
      }
    } catch {
      // Fallback for binary/opaque manifests
    }
    return {
      pageId: 'page',
      revision: 1,
      width: 900,
      height: 1280,
      layers: [],
    }
  }

  function render() {
    if (!ctx) return
    const pw = Math.max(1, Math.round(width * dpr))
    const ph = Math.max(1, Math.round(height * dpr))
    if (element.width !== pw || element.height !== ph) {
      element.width = pw
      element.height = ph
    }

    ctx.save()
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.fillStyle = `rgba(${background[0]}, ${background[1]}, ${background[2]}, ${(background[3] ?? 255) / 255})`
    ctx.fillRect(0, 0, pw, ph)

    if (activeFrame) {
      ctx.setTransform(zoom, 0, 0, zoom, tx, ty)
      const pageW = activeFrame.width || 900
      const pageH = activeFrame.height || 1280

      // Page drop shadow & paper background
      ctx.save()
      ctx.shadowColor = 'rgba(0, 0, 0, 0.16)'
      ctx.shadowBlur = 24
      ctx.shadowOffsetY = 6
      ctx.fillStyle = '#ffffff'
      ctx.fillRect(0, 0, pageW, pageH)
      ctx.restore()

      // Draw manga page artwork panels
      drawMangaArtwork(ctx, pageW, pageH, activeFrame.pageIndex || 0)

      // Draw layers from manifest
      const layers = Array.isArray(activeFrame.layers) ? activeFrame.layers : []
      for (const layer of layers) {
        if (!layer.visibility?.visible) continue
        const previewAlpha = opacityPreviews.get(layer.id)
        const opacity = previewAlpha !== undefined && previewAlpha !== null ? previewAlpha : (layer.visibility?.opacity ?? 1)
        ctx.save()
        ctx.globalAlpha = Math.max(0, Math.min(1, opacity))

        if (layer.type === 'text' && layer.geometry?.points?.length >= 4) {
          const override = transformPreview.get(layer.id)
          const pts = layer.geometry.points
          const xs = pts.map((p) => p.x)
          const ys = pts.map((p) => p.y)
          const x = override ? override.x : Math.min(...xs)
          const y = override ? override.y : Math.min(...ys)
          const w = override ? override.width : Math.max(40, Math.max(...xs) - x)
          const h = override ? override.height : Math.max(24, Math.max(...ys) - y)
          const display = layer.content?.translation?.text || layer.content?.source?.text || ''
          if (display) {
            // Clean white bubble fill behind translated text
            ctx.fillStyle = '#ffffff'
            ctx.beginPath()
            ctx.roundRect(x - 6, y - 6, w + 12, h + 12, 12)
            ctx.fill()

            const fontSize = layer.typography?.size || 22
            ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, sans-serif`
            const color = layer.typography?.color || [24, 24, 27, 255]
            ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${(color[3] ?? 255) / 255})`
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'
            wrapText(ctx, display, x + w / 2, y + h / 2, w - 8, fontSize * 1.25)
          }
        }
        ctx.restore()
      }

      if (activeStroke && activeStroke.points.length > 0) {
        ctx.save()
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        ctx.lineWidth = activeStroke.diameter || 24
        const c = activeStroke.color || [216, 91, 106, 200]
        ctx.strokeStyle =
          activeStroke.kind === 'inpaint'
            ? 'rgba(216, 91, 106, 0.55)'
            : activeStroke.kind === 'erase'
              ? 'rgba(255, 255, 255, 0.9)'
              : `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${(c[3] ?? 255) / 255})`
        ctx.beginPath()
        const [first, ...rest] = activeStroke.points
        ctx.moveTo(first.x, first.y)
        for (const pt of rest) ctx.lineTo(pt.x, pt.y)
        ctx.stroke()
        ctx.restore()
      }
    }

    ctx.restore()
  }

  return {
    resize(w, h, nextDpr, bg) {
      width = w
      height = h
      dpr = nextDpr || 1
      background = Array.from(bg)
      render()
    },
    setView(nextZoom, nextTx, nextTy) {
      zoom = nextZoom
      tx = nextTx
      ty = nextTy
      render()
    },
    stageManifest(manifest) {
      const token = nextToken++
      stagedFrames.set(token, parseManifest(manifest))
      return { token, missing: [] }
    },
    async installResource() {
      return undefined
    },
    activateFrame(token) {
      const frame = stagedFrames.get(token)
      if (!frame) return false
      activeFrame = frame
      render()
      return true
    },
    activatePage(page, expectedRevision) {
      const cached = cachedPages.get(page)
      if (!cached || BigInt(cached.revision ?? 0) !== expectedRevision) return false
      activeFrame = cached
      render()
      return true
    },
    cacheFrame(token) {
      const frame = stagedFrames.get(token)
      if (!frame) return false
      if (frame.pageId) cachedPages.set(frame.pageId, frame)
      return true
    },
    clear() {
      activeFrame = null
      stagedFrames.clear()
      cachedPages.clear()
      opacityPreviews.clear()
      transformPreview.clear()
      render()
    },
    previewOpacity(elementId, opacity) {
      if (opacity === null) opacityPreviews.delete(elementId)
      else opacityPreviews.set(elementId, opacity)
      render()
    },
    beginTransform(elements) {
      transformPreview = new Map(elements.map((item) => [item.element, item.frame]))
      render()
    },
    updateTransform(_sequence, elements) {
      transformPreview = new Map(elements.map((item) => [item.element, item.frame]))
      render()
    },
    finishTransform() {
      transformPreview.clear()
      render()
    },
    cancelTransform() {
      transformPreview.clear()
      render()
    },
    beginStroke(kind, layer, point, diameter, color) {
      activeStroke = {
        kind,
        layer,
        points: [point],
        diameter,
        color: Array.from(color),
      }
      render()
    },
    extendStroke(points) {
      if (!activeStroke) return
      activeStroke.points.push(...points)
      render()
    },
    finishStroke() {
      activeStroke = null
      render()
    },
    cancelStroke() {
      activeStroke = null
      render()
    },
    async sampleColor() {
      return new Uint8Array([255, 255, 255, 255])
    },
    setDeviceLostCallback() {},
    dispose() {},
    free() {},
  }
}

function drawMangaArtwork(ctx, w, h, variant) {
  ctx.save()
  ctx.strokeStyle = '#18181b'
  ctx.lineWidth = 4

  // Outer panel border
  const pad = 48
  const innerW = w - pad * 2
  const topH = Math.round((h - pad * 2) * 0.54)
  const botY = pad + topH + 22
  const botH = h - pad - botY

  // Top panel background
  const topGrad = ctx.createLinearGradient(pad, pad, pad + innerW, pad + topH)
  topGrad.addColorStop(0, variant % 2 === 0 ? '#fdf2f4' : '#f3f4f6')
  topGrad.addColorStop(1, '#e5e7eb')
  ctx.fillStyle = topGrad
  ctx.fillRect(pad, pad, innerW, topH)
  ctx.strokeRect(pad, pad, innerW, topH)

  // Decorative manga speedlines in top panel
  ctx.save()
  ctx.beginPath()
  ctx.rect(pad, pad, innerW, topH)
  ctx.clip()
  ctx.strokeStyle = 'rgba(24, 24, 27, 0.08)'
  ctx.lineWidth = 1.5
  const cx = pad + innerW * 0.48
  const cy = pad + topH * 0.55
  for (let i = 0; i < 36; i++) {
    const angle = (i / 36) * Math.PI * 2
    ctx.beginPath()
    ctx.moveTo(cx + Math.cos(angle) * 110, cy + Math.sin(angle) * 110)
    ctx.lineTo(cx + Math.cos(angle) * 520, cy + Math.sin(angle) * 520)
    ctx.stroke()
  }

  // Stylized character silhouette in top panel
  ctx.fillStyle = '#27272a'
  ctx.beginPath()
  ctx.arc(cx, cy - 30, 56, 0, Math.PI * 2)
  ctx.fill()
  ctx.beginPath()
  ctx.roundRect(cx - 82, cy + 32, 164, 220, 44)
  ctx.fill()
  ctx.restore()

  // Speech bubble 1 (top right)
  drawSpeechBubble(ctx, pad + innerW - 255, pad + 38, 215, 220)

  // Bottom left & right panels
  const leftW = Math.round(innerW * 0.48)
  const rightX = pad + leftW + 20
  const rightW = innerW - leftW - 20

  ctx.fillStyle = '#f8fafc'
  ctx.fillRect(pad, botY, leftW, botH)
  ctx.strokeRect(pad, botY, leftW, botH)

  ctx.fillStyle = '#f4f4f5'
  ctx.fillRect(rightX, botY, rightW, botH)
  ctx.strokeRect(rightX, botY, rightW, botH)

  // Speech bubble 2 (bottom left)
  drawSpeechBubble(ctx, pad + 26, botY + 34, 200, 175)
  // Speech bubble 3 (bottom right)
  drawSpeechBubble(ctx, rightX + rightW - 224, botY + 46, 194, 185)

  ctx.restore()
}

function drawSpeechBubble(ctx, x, y, w, h) {
  ctx.save()
  ctx.fillStyle = '#ffffff'
  ctx.strokeStyle = '#18181b'
  ctx.lineWidth = 3
  ctx.beginPath()
  ctx.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2)
  ctx.fill()
  ctx.stroke()
  ctx.restore()
}

function wrapText(ctx, text, cx, cy, maxWidth, lineHeight) {
  const words = String(text).split(/\s+/)
  const lines = []
  let current = ''
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word
    if (ctx.measureText(candidate).width > maxWidth && current) {
      lines.push(current)
      current = word
    } else {
      current = candidate
    }
  }
  if (current) lines.push(current)
  const startY = cy - ((lines.length - 1) * lineHeight) / 2
  for (let i = 0; i < lines.length; i++) {
    ctx.fillText(lines[i], cx, startY + i * lineHeight)
  }
}
