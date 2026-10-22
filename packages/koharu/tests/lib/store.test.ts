import { describe, expect, it, vi } from 'vitest'

import { receiveDownload, receiveResources, useKoharuStore } from '@/lib/store'

describe('editor state', () => {
  it('retains failed downloads until they are dismissed', () => {
    useKoharuStore.setState({ downloads: {} })
    receiveDownload({
      state: 'failed',
      id: 9,
      name: 'weights.bin',
      completed: 0,
      total: 0,
      error: 'network unavailable',
    })
    expect(useKoharuStore.getState().downloads[9]).toMatchObject({
      state: 'failed',
      name: 'weights.bin',
    })

    useKoharuStore.getState().dismissDownload(9)
    expect(useKoharuStore.getState().downloads[9]).toBeUndefined()
  })

  it('defaults paint brush color to pure white (#FFFFFF)', () => {
    expect(useKoharuStore.getState().brush.color).toBe('#FFFFFF')
  })

  it('updates the resource monitor only when what it shows changes, twice a second at most', () => {
    vi.useFakeTimers()
    try {
      const sample = (cpu: number) => ({
        process_memory: 2e9,
        system_memory: 16e9,
        process_cpu: cpu,
        devices: [],
      })
      useKoharuStore.setState({ resources: null })
      receiveResources(sample(5))
      const first = useKoharuStore.getState().resources
      expect(first).toMatchObject({ process_cpu: 5 })

      vi.advanceTimersByTime(100)
      receiveResources(sample(9))
      expect(useKoharuStore.getState().resources).toBe(first)

      vi.advanceTimersByTime(500)
      receiveResources(sample(5.2))
      expect(useKoharuStore.getState().resources).toBe(first)
      receiveResources(sample(9))
      expect(useKoharuStore.getState().resources).toMatchObject({ process_cpu: 9 })
    } finally {
      vi.useRealTimers()
    }
  })
})
